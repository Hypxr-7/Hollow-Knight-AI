import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import BehavioralCloningNet, HollowKnightDataset, load_model_for_inference
import pandas as pd

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook for gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx, other_features):
        self.model.zero_grad()
        output = self.model(x, other_features)
        
        # Target for backprop
        target = output[:, class_idx]
        target.backward()

        # Global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weighted combination of activation maps
        activation = self.activations[0]
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activation, dim=0).cpu().detach()
        
        # ReLU on top
        heatmap = F.relu(heatmap)
        
        # Normalize
        heatmap = heatmap.numpy()
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        
        return heatmap

def apply_heatmap(image_path, heatmap, alpha=0.4):
    original_img = cv2.imread(image_path)
    
    # Resize heatmap to the dimensions of the original image
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap_colored * alpha + original_img * (1 - alpha)
    return np.uint8(superimposed_img)

def main():
    print("Starting Grad-CAM explainability script...")
    
    MODEL_DIR = 'model/ensemble_1'
    DATA_DIR = r"C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\HKData"
    OUTPUT_DIR = 'evaluation_results/explainability'
    
    if not os.path.exists(MODEL_DIR):
        # Fallback to finding any ensemble directory
        available_models = [d for d in os.listdir('model') if d.startswith('ensemble_')]
        if not available_models:
            print("No models found.")
            return
        MODEL_DIR = os.path.join('model', available_models[0])
    
    print(f"Loading model from {MODEL_DIR}...")
    model, info = load_model_for_inference(MODEL_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # We want to look at the last convolutional layer (conv2)
    # Based on the architecture in train.py: conv1 -> pool -> conv2 -> pool -> flatten
    grad_cam = GradCAM(model, model.conv2)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Data
    csv_files = [f for f in os.listdir(DATA_DIR) if f.startswith('hk_actions_') and f.endswith('.csv')]
    if not csv_files:
        print("No data found.")
        return
        
    latest_csv = max(csv_files)
    session_id = latest_csv.replace('hk_actions_', '').replace('.csv', '')
    frames_dir = os.path.join(DATA_DIR, f'frames_{session_id}')
    
    df = pd.read_csv(os.path.join(DATA_DIR, latest_csv))
    df['frame_path'] = df['frame_id'].apply(lambda x: os.path.join(frames_dir, f"frame_{x:06d}.png"))
    
    # Dataset
    image_size = tuple(info['image_shape'][2:0:-1]) # (W, H)
    dataset = HollowKnightDataset(df, image_size=image_size)
    
    action_columns = ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']
    
    print("\nGenerating explanations for sample frames...")
    
    # Find one positive sample for each action
    for action_idx, action_name in enumerate(action_columns):
        positive_samples = df[df[action_name] == 1]
        
        if len(positive_samples) == 0:
            print(f"No samples found for action: {action_name}")
            continue
            
        # Pick a random sample
        sample_row = positive_samples.sample(1).iloc[0]
        idx = sample_row.name # Original index in df might be different if filtered, but dataset handles it by index
        
        # We need the index in the *dataset*, so we find it by matching frame_id if possible, 
        # or just iterating. Dataset filters invalid frames, so indices shift.
        # Let's just search the dataset for this frame path.
        dataset_idx = -1
        for i in range(len(dataset)):
            if dataset.data.iloc[i]['frame_path'] == sample_row['frame_path']:
                dataset_idx = i
                break
        
        if dataset_idx == -1: continue

        img_tensor, other_features, actions = dataset[dataset_idx]
        img_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dim
        other_features = other_features.unsqueeze(0).to(device)
        
        # Generate Heatmap
        heatmap = grad_cam(img_tensor, action_idx, other_features)
        
        # Overlay on original image
        original_frame_path = sample_row['frame_path']
        vis_img = apply_heatmap(original_frame_path, heatmap)
        
        save_path = os.path.join(OUTPUT_DIR, f'gradcam_{action_name}.png')
        cv2.imwrite(save_path, vis_img)
        print(f"Saved explanation for '{action_name}' to {save_path}")

    print(f"\nDone! Explanations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
