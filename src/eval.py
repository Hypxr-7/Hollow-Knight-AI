import os
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from scipy.stats import probplot
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader

# Import classes
from train import BehavioralCloningNet, HollowKnightDataset, load_model_for_inference, MetaLearner

warnings.filterwarnings('ignore')

# --- Evaluation and Plotting Functions ---

def get_predictions(model_base_dir, master_df, device='cpu'):
    """
    Gets predictions using the Meta-Learner ensemble architecture.
    """
    print(f"Loading models from {model_base_dir}...")
    
    expert_names = ['global', 'movement', 'jump_dash', 'combat']
    experts = {}
    max_n_frames = 1
    total_expert_output_dim = 0
    
    # Load Experts
    for name in expert_names:
        model_dir = os.path.join(model_base_dir, f"ensemble_{name}")
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Expert model not found at {model_dir}")
            
        model, info = load_model_for_inference(model_dir)
        model.to(device)
        model.eval()
        experts[name] = {'model': model, 'info': info}
        
        total_expert_output_dim += len(info.get('action_columns', []))
        
        n_frames = info.get('n_frames', 1)
        if 'image_shape' in info and len(info['image_shape']) == 3:
            n_frames = max(n_frames, info['image_shape'][0])
        max_n_frames = max(max_n_frames, n_frames)

    # Load Meta-Learner Info
    meta_info_path = os.path.join(model_base_dir, 'meta_info.json')
    if not os.path.exists(meta_info_path):
            raise FileNotFoundError(f"Meta info not found at {meta_info_path}")
    
    with open(meta_info_path, 'r') as f:
        meta_info = json.load(f)
        
    meta_features_list = meta_info.get('meta_features', [])
    meta_input_dim = total_expert_output_dim + len(meta_features_list)
    print(f"Meta-Learner Input Dim: {meta_input_dim} (Experts: {total_expert_output_dim} + Context: {len(meta_features_list)})")

    # Load Meta-Learner
    meta_path = os.path.join(model_base_dir, 'meta_learner.pth')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta-Learner model not found at {meta_path}")
        
    meta_learner = MetaLearner(input_dim=meta_input_dim, output_dim=5)
    meta_learner.load_state_dict(torch.load(meta_path, map_location=device))
    meta_learner.to(device)
    meta_learner.eval()
    
    print(f"Models loaded. Max frames needed: {max_n_frames}")

    # Prepare Dataset
    # We load ALL features into the dataframe first
    # This dataset returns "all available features" in a tensor, but we also need column mapping
    target_size = experts['global']['info']['image_shape'][1:] # H, W
    image_size = (target_size[1], target_size[0]) # W, H for PIL
    
    # We need a custom dataset that returns the DataFrame row or a Dict
    # because standard HollowKnightDataset returns a fixed feature tensor.
    # We will subclass and return the raw indices to look up in DF
    
    class EvalDataset(HollowKnightDataset):
        def __getitem__(self, idx):
            img, _, actions = super().__getitem__(idx) # Ignore standard feature tensor
            frame_id = self.data.iloc[idx]['frame_id']
            # Return index to look up row in master_df efficiently
            return img, idx, actions, frame_id

    # We assume master_df is already engineered with ALL columns
    feature_columns_all = list(set([col for exp in experts.values() for col in exp['info']['feature_columns']] + meta_features_list))
    
    # Pass a dummy feature column list to init, we won't use the standard output
    eval_dataset = EvalDataset(data_df=master_df.copy(), image_size=image_size, n_frames=max_n_frames, feature_columns=['x_position']) # Dummy
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []
    all_frame_ids = []
    all_embeddings = [] 

    # Pre-extract all features to numpy for fast lookup
    # Rows: Samples, Cols: Features
    # We need a map from "Feature Name" to "Column Index"
    df_features = master_df[feature_columns_all].fillna(0.0).astype(np.float32)
    feature_map = {name: i for i, name in enumerate(feature_columns_all)}
    feature_matrix = df_features.values
    feature_matrix_torch = torch.tensor(feature_matrix, device=device)

    with torch.no_grad():
        for images, indices, labels, frame_ids in tqdm(eval_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            indices = indices.to(device)
            
            # Get the feature rows for this batch
            batch_features_all = feature_matrix_torch[indices]
            
            expert_outputs = []
            global_embedding = None

            for name in expert_names:
                expert = experts[name]
                model = expert['model']
                info = expert['info']
                
                # Select specific columns for this expert
                required_cols = info['feature_columns']
                col_indices = [feature_map[c] for c in required_cols]
                expert_features = batch_features_all[:, col_indices]
                
                # Get embeddings from global expert for visualization
                if name == 'global':
                    x = model.pool(torch.nn.functional.relu(model.bn1(model.conv1(images))))
                    x = model.pool(torch.nn.functional.relu(model.bn2(model.conv2(x))))
                    x = x.view(x.size(0), -1)
                    combined = torch.cat([x, expert_features], dim=1)
                    embedding = torch.nn.functional.relu(model.bn3(model.fc1(combined)))
                    global_embedding = embedding.cpu().numpy()
                
                logits = model(images, expert_features)
                expert_outputs.append(logits)
            
            # Construct Meta Input
            expert_logits_cat = torch.cat(expert_outputs, dim=1)
            
            meta_col_indices = [feature_map[c] for c in meta_features_list]
            meta_features_batch = batch_features_all[:, meta_col_indices]
            
            meta_input = torch.cat([expert_logits_cat, meta_features_batch], dim=1)
            
            final_logits = meta_learner(meta_input)
            probs = torch.sigmoid(final_logits)
            
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_frame_ids.extend(frame_ids.numpy())
            if global_embedding is not None:
                all_embeddings.append(global_embedding)

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.array(all_frame_ids),
        np.concatenate(all_embeddings)
    )

def plot_class_distribution(df, action_columns, output_dir):
    plt.figure(figsize=(10, 6))
    action_counts = df[action_columns].sum().sort_values(ascending=False)
    sns.barplot(x=action_counts.index, y=action_counts.values)
    plt.title('Class Distribution of Single Key Presses')
    plt.ylabel('Number of Frames')
    plt.xlabel('Action')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_class_distribution_single_key.png'))
    plt.close()

    plt.figure(figsize=(12, 7))
    action_combos = df[action_columns].value_counts().head(10)
    combo_labels = [" + ".join([col for col, val in zip(action_columns, combo) if val]) or "No Action" for combo, count in action_combos.items()]
    sns.barplot(x=action_combos.values, y=combo_labels)
    plt.title('Top 10 Key Combinations')
    plt.xlabel('Number of Frames')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_top_10_key_combinations.png'))
    plt.close()

def plot_violin(true_labels, pred_probs, action_columns, output_dir):
    all_data = []
    for i, action in enumerate(action_columns):
        for true_label, pred_prob in zip(true_labels[:, i], pred_probs[:, i]):
            all_data.append({
                'Action': action,
                'True Label': 'True 1' if true_label == 1 else 'True 0',
                'Predicted Probability': pred_prob
            })
    df_combined = pd.DataFrame(all_data)

    plt.figure(figsize=(15, 8))
    sns.violinplot(
        data=df_combined,
        x='Action',
        y='Predicted Probability',
        hue='True Label',
        split=True,
        inner='quartile',
        palette={'True 0': 'skyblue', 'True 1': 'lightcoral'}
    )
    plt.title('Predicted Probability Distribution by Action and True Label')
    plt.ylabel('Predicted Probability')
    plt.xlabel('Action')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_violin_plots_grouped_split.png'))
    plt.close()

def plot_calibration_curve(true_labels, pred_probs, action_columns, output_dir):
    for i, action in enumerate(action_columns):
        plt.figure(figsize=(8, 8))
        prob_true, prob_pred = calibration_curve(true_labels[:, i], pred_probs[:, i], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=action)
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Curve for "{action}"')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'4_calibration_curve_{action}.png'))
        plt.close()

def plot_confusion_matrix(true_labels, pred_probs, action_columns, output_dir):
    preds = (pred_probs > 0.5).astype(int)
    for i, action in enumerate(action_columns):
        cm = confusion_matrix(true_labels[:, i], preds[:, i])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix for "{action}"')
        plt.savefig(os.path.join(output_dir, f'5_confusion_matrix_{action}.png'))
        plt.close()
        
def plot_pair_plots(pred_probs, action_columns, output_dir):
    """
    Plots pair plots. Uses hexbin-style 2D histograms for off-diagonal elements
    to handle large data density better than scatter plots.
    """
    df = pd.DataFrame(pred_probs, columns=action_columns)
    
    # Use 'hist' kind which creates 2D histograms (binned plots), similar to hexbin but square bins
    # This answers the user's request for hexbin-like plotting for pairs
    g = sns.pairplot(df.sample(n=min(5000, len(df))), kind='hist', diag_kind='kde', corner=True)
    g.fig.suptitle('Pair Plot of Predicted Probabilities (Density)', y=1.02)
    plt.savefig(os.path.join(output_dir, '6_pair_plot_hexbin_style.png'))
    plt.close()

def plot_roc_pr_curves(true_labels, pred_probs, action_columns, output_dir):
    for i, action in enumerate(action_columns):
        fpr, tpr, _ = roc_curve(true_labels[:, i], pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for "{action}"')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, f'7_roc_curve_{action}.png'))
        plt.close()

        precision, recall, _ = precision_recall_curve(true_labels[:, i], pred_probs[:, i])
        avg_precision = average_precision_score(true_labels[:, i], pred_probs[:, i])
        plt.figure(figsize=(8, 6))
        plt.step(recall, precision, where='post', label=f'AP={avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for "{action}"')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(output_dir, f'8_pr_curve_{action}.png'))
        plt.close()

def plot_qq(pred_probs, action_columns, output_dir):
    for i, action in enumerate(action_columns):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        probplot(pred_probs[:, i], dist="norm", plot=axes[0])
        axes[0].set_title(f'Q-Q Plot vs Normal for "{action}"')
        probplot(pred_probs[:, i], dist="uniform", plot=axes[1])
        axes[1].set_title(f'Q-Q Plot vs Uniform for "{action}"')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'9_qq_plot_{action}.png'))
        plt.close()

def plot_kde(true_labels, pred_probs, action_columns, output_dir):
    for i, action in enumerate(action_columns):
        plt.figure(figsize=(10, 6))
        data = pd.DataFrame({'True Label': true_labels[:, i], 'Predicted Probability': pred_probs[:, i]})
        sns.kdeplot(data=data, x='Predicted Probability', hue='True Label', fill=True, common_norm=False)
        plt.title(f'Class-Conditional Density for "{action}"')
        plt.savefig(os.path.join(output_dir, f'10_kde_plot_{action}.png'))
        plt.close()

def visualize_similar_images(true_labels, embeddings, frame_ids, frame_id_to_path, action_columns, output_dir, n_pairs=5, similarity_thresholds=[0.9, 0.8, 0.7]):
    for threshold in similarity_thresholds:
        print(f"Finding similar images with >{threshold*100}% similarity...")
        pairs_found = 0
        
        # Calculate similarity on a subset if dataset is huge to avoid OOM
        subset_size = min(5000, len(embeddings))
        indices_subset = np.random.choice(len(embeddings), subset_size, replace=False)
        embeddings_subset = embeddings[indices_subset]
        labels_subset = true_labels[indices_subset]
        frame_ids_subset = frame_ids[indices_subset]

        sim_matrix = cosine_similarity(embeddings_subset)
        np.fill_diagonal(sim_matrix, 0)
        
        # Find upper triangle indices
        indices = np.triu_indices_from(sim_matrix, k=1)
        
        # Filter by threshold first to reduce sorting cost
        mask = sim_matrix[indices] > threshold
        if not np.any(mask):
            continue
            
        filtered_sims = sim_matrix[indices][mask]
        filtered_indices = (indices[0][mask], indices[1][mask])
        
        # Sort by similarity desc
        sorted_order = np.argsort(-filtered_sims)
        
        for idx in sorted_order:
            if pairs_found >= n_pairs: break
            i, j = filtered_indices[0][idx], filtered_indices[1][idx]
            
            label_i, label_j = labels_subset[i], labels_subset[j]

            # Check if labels are different
            if not np.array_equal(label_i, label_j):
                pairs_found += 1
                
                frame_id_i, frame_id_j = frame_ids_subset[i], frame_ids_subset[j]
                
                path_i = frame_id_to_path.get(frame_id_i)
                path_j = frame_id_to_path.get(frame_id_j)
                
                if path_i and path_j and os.path.exists(path_i) and os.path.exists(path_j):
                    img_i, img_j = cv2.imread(path_i), cv2.imread(path_j)
                    
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    
                    actions_i = " + ".join([col for k, col in enumerate(action_columns) if label_i[k]]) or "No Action"
                    axes[0].imshow(cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB))
                    axes[0].set_title(f'Frame {frame_id_i}\nActions: {actions_i}')
                    axes[0].axis('off')

                    actions_j = " + ".join([col for k, col in enumerate(action_columns) if label_j[k]]) or "No Action"
                    axes[1].imshow(cv2.cvtColor(img_j, cv2.COLOR_BGR2RGB))
                    axes[1].set_title(f'Frame {frame_id_j}\nActions: {actions_j}')
                    axes[1].axis('off')
                    
                    plt.suptitle(f'Similarity: {sim_matrix[i, j]:.2f} (> {threshold*100}%) with Different Labels')
                    plt.savefig(os.path.join(output_dir, f'11_similar_images_{threshold*100}pct_{pairs_found}.png'))
                    plt.close()

def main():
    print("Starting evaluation script...")
    
    DATA_DIR = r"C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\HKData"
    MODEL_DIR = 'model' # New model directory
    OUTPUT_DIR = 'evaluation_results'
    EVAL_ON_ALL_DATA = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    action_columns = ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']

    csv_files = [f for f in os.listdir(DATA_DIR) if f.startswith('hk_actions_') and f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found.")
        return

    data_to_load = csv_files if EVAL_ON_ALL_DATA else [max(csv_files)]
    
    master_df = pd.concat([
        pd.read_csv(os.path.join(DATA_DIR, csv_file)).assign(
            frame_path=lambda df, sid=csv_file.replace('hk_actions_', '').replace('.csv', ''): df['frame_id'].apply(
                lambda x: os.path.join(DATA_DIR, f'frames_{sid}', f"frame_{x:06d}.png")
            )
        )
        for csv_file in data_to_load if os.path.exists(os.path.join(DATA_DIR, f'frames_{csv_file.replace("hk_actions_", "").replace(".csv", "")}'))
    ], ignore_index=True)

    if master_df.empty:
        print("No data could be loaded for evaluation."); return
        
    try:
        true_labels, pred_probs, frame_ids, embeddings = get_predictions(MODEL_DIR, master_df, device=device)
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    print(f"Got predictions for {len(true_labels)} samples.")
    
    temp_dataset = HollowKnightDataset(data_df=master_df.copy(), image_size=None)
    frame_id_to_path = pd.Series(temp_dataset.data.frame_path.values, index=temp_dataset.data.frame_id).to_dict()

    print("Generating plots...")
    plot_class_distribution(temp_dataset.data, action_columns, OUTPUT_DIR)
    print("1, 2. Class distribution plots generated.")
    plot_violin(true_labels, pred_probs, action_columns, OUTPUT_DIR)
    print("3. Violin plots generated.")
    plot_calibration_curve(true_labels, pred_probs, action_columns, OUTPUT_DIR)
    print("4. Calibration curves generated.")
    plot_confusion_matrix(true_labels, pred_probs, action_columns, OUTPUT_DIR)
    print("5. Confusion matrices generated.")
    plot_pair_plots(pred_probs, action_columns, OUTPUT_DIR)
    print("6. Pair plots (Hexbin/Hist style) generated.")
    plot_roc_pr_curves(true_labels, pred_probs, action_columns, OUTPUT_DIR)
    print("7, 8. ROC and Precision-Recall curves generated.")
    plot_qq(pred_probs, action_columns, OUTPUT_DIR)
    print("9. Q-Q plots generated.")
    plot_kde(true_labels, pred_probs, action_columns, OUTPUT_DIR)
    print("10. KDE plots generated.")
    visualize_similar_images(true_labels, embeddings, frame_ids, frame_id_to_path, action_columns, OUTPUT_DIR)
    print("11. Similar image visualizations generated.")
    print(f"\nEvaluation complete! All results saved in '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    main()