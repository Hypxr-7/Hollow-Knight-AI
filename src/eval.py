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

# Import the correct, up-to-date classes and functions from train.py
from train import BehavioralCloningNet, ResNetLSTMBehavioralCloningNet, HollowKnightDataset, load_model_for_inference, engineer_features

warnings.filterwarnings('ignore')

# --- Evaluation and Plotting Functions ---

class GradCAM:
    """
    Grad-CAM implementation for ResNet-based BehavioralCloningNet.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, other_features, class_idx):
        self.model.eval() 
        
        # Forward pass
        output = self.model(x, other_features)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, output.size(-1)).zero_().to(x.device)
        one_hot_output[0][class_idx] = 1
        
        # Backward pass
        output.backward(gradient=one_hot_output)
        
        # Global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activations by gradients
        activations = self.activations.detach().clone()
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU on top
        heatmap = np.maximum(heatmap.cpu(), 0)
        
        # Normalize
        heatmap /= torch.max(heatmap)
        
        return heatmap.numpy()

def generate_gradcam_visualizations(model, dataset, action_columns, output_dir, device='cpu', num_samples=5, scaler=None):
    """
    Generates Grad-CAM heatmaps for specific actions.
    """
    print("Generating Grad-CAM visualizations...")
    
    # Target the last convolutional layer of the ResNet backbone
    # model.resnet.layer4 is usually the last block in ResNet18
    target_layer = model.resnet.layer4[-1].conv2
    grad_cam = GradCAM(model, target_layer)
    
    model.to(device)
    model.eval()
    
    # We want to find samples where the model strongly predicts an action
    # Let's pick 'attacking' and 'jumping'
    target_actions = ['attacking', 'jumping']
    
    for action_name in target_actions:
        if action_name not in action_columns: continue
        
        action_idx = action_columns.index(action_name)
        samples_found = 0
        
        # Shuffle dataset to get random samples
        indices = np.random.permutation(len(dataset))
        
        for idx in indices:
            if samples_found >= num_samples: break
            
            img, other_features, label, frame_id = dataset[idx]
            
            # Apply scaling to other_features if scaler is provided
            if scaler:
                # other_features is a tensor, scaler expects numpy (1, N)
                feat_np = other_features.numpy().reshape(1, -1)
                feat_scaled = scaler.transform(feat_np)[0]
                other_features = torch.FloatTensor(feat_scaled)

            # Check if this sample actually has the label
            if label[action_idx] == 1:
                img_tensor = img.unsqueeze(0).to(device)
                feat_tensor = other_features.unsqueeze(0).to(device)
                
                # Get model prediction to confirm it's confident
                with torch.no_grad():
                    output = model(img_tensor, feat_tensor)
                    prob = torch.sigmoid(output)[0][action_idx].item()
                
                if prob > 0.7: # Only visualize high confidence TP
                    samples_found += 1
                    
                    # Generate Heatmap
                    heatmap = grad_cam(img_tensor, feat_tensor, action_idx)
                    
                    # Process original image for display
                    # Take the last frame from the stack
                    last_frame_tensor = img[-1, :, :]
                    original_img = (last_frame_tensor.cpu().numpy() * 255).astype(np.uint8)
                    original_img_color = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
                    
                    # Resize heatmap to match image size
                    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    
                    # Overlay
                    superimposed_img = cv2.addWeighted(original_img_color, 0.6, heatmap, 0.4, 0)
                    
                    plt.figure(figsize=(10, 5))
                    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
                    plt.title(f"Grad-CAM: {action_name} (Prob: {prob:.2f})\nFrame: {frame_id}")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'12_gradcam_{action_name}_{samples_found}.png'))
                    plt.close()

def get_predictions(models_with_info, master_df, device='cpu'):
    """
    Gets predictions from an ensemble of models, handling heterogeneous input sizes.
    """
    all_model_preds = []
    all_model_embeddings = []
    
    final_labels = None
    final_frame_ids = None

    class TempEvalDataset(HollowKnightDataset):
        def __init__(self, data_df, image_size, n_frames, sequence_length, feature_columns):
            super().__init__(data_df=data_df, image_size=image_size, n_frames=n_frames, sequence_length=sequence_length, feature_columns=feature_columns)

        def __getitem__(self, idx):
            img, other_features, actions = super().__getitem__(idx)
            # If sequence, we take the last frame_id of the sequence for reference
            start_idx = self.valid_indices[idx]
            frame_id = self.data.iloc[start_idx + self.sequence_length - 1]['frame_id']
            return img, other_features, actions, frame_id

    for i, (model, info, scaler) in enumerate(models_with_info):
        print(f"--- Processing Model {i+1}/{len(models_with_info)} ---")
        model.to(device)
        model.eval()

        # Handle legacy models or different shapes
        n_frames = info.get('n_frames', 1)
        sequence_length = info.get('sequence_length', 1)
        if 'image_shape' in info and len(info['image_shape']) == 3:
            n_frames = max(n_frames, info['image_shape'][0])
            
        target_h = info['image_shape'][1]
        target_w = info['image_shape'][2]
        image_size = (target_w, target_h) # PIL uses (W, H)
        feature_columns = info.get('feature_columns', ['x_position', 'y_position', 'enemy_x', 'enemy_y'])

        print(f"Using image size: {image_size}, n_frames: {n_frames}, seq_len: {sequence_length}")

        temp_dataset = TempEvalDataset(
            data_df=master_df.copy(), 
            image_size=image_size, 
            n_frames=n_frames, 
            sequence_length=sequence_length,
            feature_columns=feature_columns
        )
        temp_loader = DataLoader(temp_dataset, batch_size=64, shuffle=False)
        
        current_model_labels, current_model_preds, current_model_frame_ids, current_model_embeddings = [], [], [], []

        with torch.no_grad():
            for images, other_features, labels, frame_ids in tqdm(temp_loader, desc=f"Model {i+1} Predictions"):
                images, labels = images.to(device), labels.to(device)
                
                if scaler:
                    # Scaling needs to be applied carefully.
                    # other_features is (B, S, F) or (B, F)
                    # Scaler expects 2D.
                    original_shape = other_features.shape
                    feat_flat = other_features.view(-1, original_shape[-1]).numpy()
                    feat_scaled = scaler.transform(feat_flat)
                    other_features = torch.FloatTensor(feat_scaled).view(original_shape).to(device)
                else:
                    other_features = other_features.to(device)

                # Handle LSTM
                if isinstance(model, ResNetLSTMBehavioralCloningNet):
                    # Output is (B, S, Actions)
                    outputs, _ = model(images, other_features)
                    # We might want the predictions for the WHOLE sequence or just the last step?
                    # Usually for eval we check every step.
                    # But the Dataset yields overlapping sequences?
                    # HollowKnightDataset with seq>1 yields windowed sequences (stride 1).
                    # So for every index, we get a sequence. 
                    # The label returned by dataset is also a sequence.
                    # We can evaluate the LAST item in the sequence (most context).
                    output = outputs[:, -1, :]
                    labels = labels[:, -1, :]
                    
                    # Embedding? We can use the last hidden state or fusion output.
                    # For simplicity, let's skip embedding visualization for LSTM for now or use the LSTM output before head
                    embedding = torch.zeros(outputs.size(0), 256).to(device) # Dummy
                else:
                    # Standard ResNet
                    x = model.resnet(images) 
                    combined = torch.cat([x, other_features], dim=1)
                    embedding = torch.nn.functional.relu(model.bn1(model.fusion_fc(combined)))
                    output = model.final_fc(model.dropout(embedding))

                probabilities = torch.sigmoid(output)

                current_model_labels.append(labels.cpu().numpy())
                current_model_preds.append(probabilities.cpu().numpy())
                current_model_frame_ids.extend(frame_ids.numpy())
                current_model_embeddings.append(embedding.cpu().numpy())

        all_model_preds.append(np.concatenate(current_model_preds))
        all_model_embeddings.append(np.concatenate(current_model_embeddings))
        
        if final_labels is None:
            final_labels = np.concatenate(current_model_labels)
            final_frame_ids = np.array(current_model_frame_ids)

    avg_preds = np.mean(all_model_preds, axis=0)
    avg_embeddings = np.mean(all_model_embeddings, axis=0)

    return final_labels, avg_preds, final_frame_ids, avg_embeddings

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
    sns.pairplot(df.sample(n=min(1000, len(df))), kind='reg', diag_kind='kde')
    plt.suptitle('Pair Plot of Predicted Probabilities', y=1.02)
    plt.savefig(os.path.join(output_dir, '6_pair_plot_of_probabilities.png'))
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
        
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)
        indices = np.triu_indices_from(sim_matrix, k=1)
        sorted_indices = np.argsort(-sim_matrix[indices])

        for idx in sorted_indices:
            if pairs_found >= n_pairs: break
            i, j = indices[0][idx], indices[1][idx]
            
            if sim_matrix[i, j] > threshold:
                label_i, label_j = true_labels[i], true_labels[j]

                if not np.array_equal(label_i, label_j):
                    pairs_found += 1
                    
                    frame_id_i, frame_id_j = frame_ids[i], frame_ids[j]
                    img_i, img_j = cv2.imread(frame_id_to_path[frame_id_i]), cv2.imread(frame_id_to_path[frame_id_j])
                    
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    
                    actions_i = " + ".join([col for k, col in enumerate(action_columns) if label_i[k]]) or "No Action"
                    axes[0].imshow(cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB))
                    axes[0].set_title(f'Frame {frame_id_i}\nActions: {actions_i}')
                    axes[0].axis('off')

                    actions_j = " + ".join([col for k, col in enumerate(action_columns) if label_j[k]]) or "No Action"
                    axes[1].imshow(cv2.cvtColor(img_j, cv2.COLOR_BGR2RGB))
                    axes[1].set_title(f'Frame {frame_id_j}\nActions: {actions_j}')
                    axes[1].axis('off')
                    
                    plt.suptitle(f'Similarity: {sim_matrix[i, j]:.2f} (> {threshold*100}% ) with Different Labels')
                    plt.savefig(os.path.join(output_dir, f'11_similar_images_{threshold*100}pct_{pairs_found}.png'))
                    plt.close()

def main():
    print("Starting evaluation script...")
    
    DATA_DIR = r"C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\HKData"
    MODEL_DIR = 'model'
    OUTPUT_DIR = 'evaluation_results'
    EVAL_ON_ALL_DATA = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading ensemble for evaluation...")
    ensemble_dirs = sorted([os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if d.startswith('ensemble_')])
    if not ensemble_dirs:
        print(f"No ensemble models found in {MODEL_DIR}. Exiting.")
        return

    models_with_info = []
    for model_dir in ensemble_dirs:
        model, info, scaler = load_model_for_inference(model_dir)
        models_with_info.append((model, info, scaler))

    if not models_with_info:
        print("No models were successfully loaded. Exiting.")
        return
        
    action_columns = models_with_info[0][1]['action_columns']
    feature_columns = models_with_info[0][1].get('feature_columns')
    print(f"Loaded {len(models_with_info)} models.")

    csv_files = [f for f in os.listdir(DATA_DIR) if f.startswith('hk_actions_') and f.endswith('.csv')]
    data_to_load = csv_files if EVAL_ON_ALL_DATA else [max(csv_files)]
    
    master_df = pd.concat([
        pd.read_csv(os.path.join(DATA_DIR, csv_file)).assign(
            frame_path=lambda df, sid=csv_file.replace('hk_actions_', '').replace('.csv', ''): df['frame_id'].apply(
                lambda x: os.path.join(DATA_DIR, f'frames_{sid}', f"frame_{x:06d}.png")
            )
        )
        for csv_file in data_to_load if os.path.exists(os.path.join(DATA_DIR, f'frames_{csv_file.replace("hk_actions_", "").replace(".csv", "")}') )
    ], ignore_index=True)
    
    # Pre-calculate features just like in training
    master_df = engineer_features(master_df)

    if master_df.empty:
        print("No data could be loaded for evaluation"); return
        
    true_labels, pred_probs, frame_ids, embeddings = get_predictions(models_with_info, master_df)
    print(f"Got predictions for {len(true_labels)} samples.")
    
    # Use the first model to generate Grad-CAMs
    # We need a Dataset that returns frame_id as part of item to match logic
    class EvalDataset(HollowKnightDataset):
        def __init__(self, data_df, image_size, n_frames, feature_columns):
            super().__init__(data_df=data_df, image_size=image_size, n_frames=n_frames, feature_columns=feature_columns)
        def __getitem__(self, idx):
            img, other_features, actions = super().__getitem__(idx)
            frame_id = self.data.iloc[idx]['frame_id']
            return img, other_features, actions, frame_id
            
    first_model, first_info, first_scaler = models_with_info[0]
    n_frames = first_info.get('n_frames', 1)
    image_size = tuple(first_info['image_shape'][1:][::-1]) # (W, H)
    
    # Pre-process GradCAM Data
    df_gradcam = master_df.copy()
    
    # We need to scale the features manually for GradCAM dataset because we are creating it directly
    if first_scaler:
        # We need to find which columns correspond to the scaler
        # In train.py we fit on 'input_feature_columns'
        # In model_info, 'feature_columns' SHOULD be the input columns used
        cols_to_scale = first_info['feature_columns']
        # Check if they exist
        existing_cols = [c for c in cols_to_scale if c in df_gradcam.columns]
        if len(existing_cols) == len(cols_to_scale):
            df_gradcam[cols_to_scale] = first_scaler.transform(df_gradcam[cols_to_scale])
        else:
            print("Warning: Could not match columns for scaling in GradCAM. Skipping scaling.")

    gradcam_dataset = EvalDataset(df_gradcam, image_size=image_size, n_frames=n_frames, feature_columns=first_info['feature_columns'])
    generate_gradcam_visualizations(first_model, gradcam_dataset, action_columns, OUTPUT_DIR, scaler=None)

    print("Generating plots...")
    # Re-using a generic dataset for plotting utils that don't need scaling/tensors
    temp_dataset = HollowKnightDataset(data_df=master_df.copy(), image_size=None, feature_columns=None)
    frame_id_to_path = pd.Series(temp_dataset.data.frame_path.values, index=temp_dataset.data.frame_id).to_dict()

    plot_class_distribution(temp_dataset.data, action_columns, OUTPUT_DIR)
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
