
import os
import json
import warnings
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from scipy.stats import probplot

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# --- Model and Dataset Classes (Copied from train.py) ---

class HollowKnightDataset(Dataset):
    def __init__(self, data_df, transform=None, image_size=(80, 60)):
        self.data = data_df
        self.transform = transform
        self.image_size = image_size
        self.action_columns = ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']
        
        if 'frame_path' not in self.data.columns:
            raise ValueError("DataFrame must contain a 'frame_path' column.")

        self.valid_indices = self._get_valid_indices()
        self.data = self.data.iloc[self.valid_indices].reset_index(drop=True)
        print(f"Dataset loaded: {len(self.data)} samples with valid frames")

    def _get_valid_indices(self):
        valid_indices = []
        for idx, row in self.data.iterrows():
            if os.path.exists(row['frame_path']):
                valid_indices.append(idx)
        return valid_indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        frame_path = row['frame_path']
        image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.image_size)
        image = image.astype(np.float32) / 255.0
        
        image_features = torch.FloatTensor(image).flatten()
        
        player_x = torch.FloatTensor([row['x_position'] / 1920.0])
        player_y = torch.FloatTensor([row['y_position'] / 1080.0])
        enemy_x = torch.FloatTensor([row['enemy_x'] / 1920.0])
        enemy_y = torch.FloatTensor([row['enemy_y'] / 1080.0])
        
        features = torch.cat([image_features, player_x, player_y, enemy_x, enemy_y])
        
        actions = torch.FloatTensor([float(row[col]) for col in self.action_columns])
        
        return features, actions, row['frame_id']

class BehavioralCloningNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], output_dim=5, dropout_rate=0.3):
        super(BehavioralCloningNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def load_model_for_inference(model_dir='model'):
    with open(os.path.join(model_dir, 'model_info.json'), 'r') as f:
        model_info = json.load(f)
    
    model = BehavioralCloningNet(
        input_dim=model_info['input_dim'],
        hidden_dims=model_info['hidden_dims'],
        output_dim=model_info['output_dim']
    )
    
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location='cpu'))
    model.eval()
    return model, model_info

# --- New Evaluation and Plotting Functions ---

def get_predictions(model, data_loader, device='cpu'):
    model.to(device)
    all_features, all_labels, all_preds, all_frame_ids, all_embeddings = [], [], [], [], []

    feature_extractor = model.network[:-1]
    classifier = model.network[-1]

    with torch.no_grad():
        for features, labels, frame_ids in tqdm(data_loader, desc="Getting Predictions"):
            features, labels = features.to(device), labels.to(device)
            
            embeddings = feature_extractor(features)
            outputs = classifier(embeddings)
            probabilities = torch.sigmoid(outputs)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(probabilities.cpu().numpy())
            all_frame_ids.extend(frame_ids.numpy())
            all_embeddings.append(embeddings.cpu().numpy())

    return (np.concatenate(all_labels), np.concatenate(all_preds), 
            np.array(all_frame_ids), np.concatenate(all_embeddings),
            np.concatenate(all_features))

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
    df = pd.DataFrame(pred_probs, columns=action_columns)
    plt.figure(figsize=(15, 15))
    sns.pairplot(df.sample(n=min(1000, len(df))), kind='reg', diag_kind='kde') # a sample to make it faster
    plt.suptitle('Pair Plot of Predicted Probabilities', y=1.02)
    plt.savefig(os.path.join(output_dir, '6_pair_plot_of_probabilities.png'))
    plt.close()

def plot_roc_pr_curves(true_labels, pred_probs, action_columns, output_dir):
    for i, action in enumerate(action_columns):
        # ROC Curve
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

        # Precision-Recall Curve
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
    from sklearn.metrics.pairwise import cosine_similarity
    
    for threshold in similarity_thresholds:
        print(f"Finding similar images with >{threshold*100}% similarity...")
        pairs_found = 0
        
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0) # Exclude self-similarity

        for i in range(len(sim_matrix)):
            if pairs_found >= n_pairs:
                break
            
            # Find most similar image
            j = np.argmax(sim_matrix[i])
            
            if sim_matrix[i, j] > threshold:
                label_i = true_labels[i]
                label_j = true_labels[j]

                if not np.array_equal(label_i, label_j):
                    pairs_found += 1
                    
                    frame_id_i = frame_ids[i]
                    frame_id_j = frame_ids[j]

                    img_i = cv2.imread(frame_id_to_path[frame_id_i])
                    img_j = cv2.imread(frame_id_to_path[frame_id_j])
                    
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    
                    actions_i = " + ".join([col for k, col in enumerate(action_columns) if label_i[k]]) or "No Action"
                    axes[0].imshow(cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB))
                    axes[0].set_title(f'Frame {frame_id_i}\nActions: {actions_i}')
                    axes[0].axis('off')

                    actions_j = " + ".join([col for k, col in enumerate(action_columns) if label_j[k]]) or "No Action"
                    axes[1].imshow(cv2.cvtColor(img_j, cv2.COLOR_BGR2RGB))
                    axes[1].set_title(f'Frame {frame_id_j}\nActions: {actions_j}')
                    axes[1].axis('off')
                    
                    plt.suptitle(f'Similarity: {sim_matrix[i, j]:.2f} (> {threshold*100}%)with Different Labels')
                    plt.savefig(os.path.join(output_dir, f'11_similar_images_{threshold*100}pct_{pairs_found}.png'))
                    plt.close()

                    # Avoid re-finding the same pair
                    sim_matrix[i, j] = sim_matrix[j, i] = 0


def main():
    print("Starting evaluation script...")
    
    # --- Configuration ---
    # NOTE: Using hardcoded paths as requested
    DATA_DIR = r"C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\HKData"
    MODEL_DIR = 'model'
    OUTPUT_DIR = 'evaluation_results'
    IMAGE_SIZE = (80, 60)
    BATCH_SIZE = 64
    EVAL_ON_ALL_DATA = True # Set to True to evaluate on all data in HKData

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load Model ---
    model, model_info = load_model_for_inference(MODEL_DIR)
    action_columns = model_info['action_columns']
    print("Model loaded successfully.")

    # --- Load Dataset ---
    csv_files = [f for f in os.listdir(DATA_DIR) if f.startswith('hk_actions_') and f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        return
    
    data_to_load = []
    if EVAL_ON_ALL_DATA:
        print("Loading all data sessions for evaluation...")
        data_to_load = csv_files
    else:
        print("Loading latest data session for evaluation...")
        data_to_load.append(max(csv_files))

    all_dfs = []
    for csv_file in data_to_load:
        session_id = csv_file.replace('hk_actions_', '').replace('.csv', '')
        frames_dir = os.path.join(DATA_DIR, f'frames_{session_id}')
        csv_path = os.path.join(DATA_DIR, csv_file)

        if os.path.exists(frames_dir):
            print(f"Processing session: {session_id}")
            df = pd.read_csv(csv_path)
            df['frame_path'] = df['frame_id'].apply(lambda x: os.path.join(frames_dir, f"frame_{x:06d}.png"))
            all_dfs.append(df)
        else:
            print(f"Warning: Frames directory not found for session {session_id}, skipping.")

    if not all_dfs:
        print("No data could be loaded for evaluation.")
        return
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    dataset = HollowKnightDataset(
        data_df=master_df,
        image_size=IMAGE_SIZE
    )
    if len(dataset) == 0:
        print("No valid data found in the dataset.")
        return
        
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Get Predictions ---
    true_labels, pred_probs, frame_ids, embeddings, features = get_predictions(model, data_loader)
    print(f"Got predictions for {len(true_labels)} samples.")

    # --- Generate Plots ---
    print("Generating plots...")
    
    plot_class_distribution(dataset.data, action_columns, OUTPUT_DIR)
    print("1, 2. Class distribution plots generated.")
    
    plot_violin(true_labels, pred_probs, action_columns, OUTPUT_DIR)
    print("3. Violin plots generated.")

    # --- Get Predictions ---
    true_labels, pred_probs, frame_ids, embeddings, features = get_predictions(model, data_loader)
    print(f"Got predictions for {len(true_labels)} samples.")

    frame_id_to_path = pd.Series(dataset.data.frame_path.values, index=dataset.data.frame_id).to_dict()

    # --- Generate Plots ---
    print("Generating plots...")
    
    plot_class_distribution(dataset.data, action_columns, OUTPUT_DIR)
    print("1, 2. Class distribution plots generated.")
    
    plot_violin(true_labels, pred_probs, action_columns, OUTPUT_DIR)
    print("3. Violin plots generated.")

    plot_calibration_curve(true_labels, pred_probs, action_columns, OUTPUT_DIR)
    print("4. Calibration curves generated.")

    plot_confusion_matrix(true_labels, pred_probs, action_columns, OUTPUT_DIR)
    print("5. Confusion matrices generated.")
    
    plot_pair_plots(pred_probs, action_columns, OUTPUT_DIR)
    print("6. Pair plots of probabilities generated.")
    
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
