# Standard library imports
import os
import json
import random
import warnings

# Third-party imports
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Scikit-learn imports
from sklearn.model_selection import train_test_split

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import torch.nn.functional as F

# Warnings configuration
warnings.filterwarnings('ignore')

class HollowKnightDataset(Dataset):
    """
    Dataset for Hollow Knight behavioral cloning.
    Prepares image and positional data for the CNN model.
    """
    def __init__(self, data_df, transform=None, image_size=(80, 60)):
        self.data = data_df.copy() # Use a copy to avoid SettingWithCopyWarning
        self.transform = transform
        self.action_columns = ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']
        
        # Directly filter the dataframe to only include rows with valid frame paths
        # This is much safer than collecting and using indices after a train/test split.
        self.data = self.data[self.data['frame_path'].apply(os.path.exists)].reset_index(drop=True)

        self.image_size = image_size
        
        if not self.image_size:
            self.image_size = self._get_original_image_size()

        print(f"Dataset loaded: {len(self.data)} samples with valid frames")
        if self.image_size:
            print(f"Image size: {self.image_size}")
        else:
            print("Image size: Not determined (no valid frames found).")
        
        print("\n=== Action Distribution ===")
        for col in self.action_columns:
            count = self.data[col].sum()
            pct = (count / len(self.data)) * 100 if len(self.data) > 0 else 0
            print(f"{col:15s}: {int(count):5d} frames ({pct:.1f}%)")
        
        enemy_present = ((self.data['enemy_x'] != 0) | (self.data['enemy_y'] != 0)).sum()
        enemy_pct = (enemy_present / len(self.data)) * 100 if len(self.data) > 0 else 0
        print(f"{'enemy_present':15s}: {int(enemy_present):5d} frames ({enemy_pct:.1f}%)")
    
    def _get_original_image_size(self):
        if len(self.data) > 0:
            first_frame_path = self.data['frame_path'].iloc[0]
            if os.path.exists(first_frame_path):
                with Image.open(first_frame_path) as img:
                    return img.size # (width, height)
        return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image, ensure it's grayscale, and resize
        with Image.open(row['frame_path']).convert('L') as img:
            if self.image_size:
                img = img.resize(self.image_size, Image.Resampling.LANCZOS)
            
            if self.transform:
                img = self.transform(img)
            else:
                # Default transform to tensor and normalize
                img = transforms.ToTensor()(img)

        # Other features (player and enemy positions)
        other_features = torch.FloatTensor([
            row['x_position'] / 1920.0,
            row['y_position'] / 1080.0,
            row['enemy_x'] / 1920.0,
            row['enemy_y'] / 1080.0
        ])
        
        actions = torch.FloatTensor([float(row[col]) for col in self.action_columns])
        
        # Return image, other features, and actions separately
        return img, other_features, actions


class BehavioralCloningNet(nn.Module):
    """
    CNN for multi-label behavioral cloning.
    Processes image data with convolutional layers and combines it with positional data.
    """
    def __init__(self, image_shape=(1, 60, 80), other_features_dim=4, num_actions=5, dropout_rate=0.5):
        super(BehavioralCloningNet, self).__init__()
        
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(in_channels=image_shape[0], out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces dimensions by half

        # Calculate flattened size after conv and pooling layers
        self._conv_out_shape = self._get_conv_out_shape(image_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._conv_out_shape + other_features_dim, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_actions)

    def _get_conv_out_shape(self, shape):
        # Create a dummy tensor and pass it through the conv layers to find output shape
        with torch.no_grad():
            dummy_tensor = torch.zeros(1, *shape) # (N, C, H, W)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy_tensor))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            return int(np.prod(x.shape))

    def forward(self, image, other_features):
        # Process image through CNN
        x = self.pool(F.relu(self.bn1(self.conv1(image))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten the output of the CNN
        x = x.view(x.size(0), -1)
        
        # Concatenate with other features
        combined = torch.cat([x, other_features], dim=1)
        
        # Pass through fully connected layers
        combined = F.relu(self.bn3(self.fc1(combined)))
        combined = self.dropout(combined)
        output = self.fc2(combined) # No sigmoid, handled by BCEWithLogitsLoss
        
        return output


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def forward(self, inputs, targets):
        logpt = -self.bce_loss(inputs, targets)
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


class BehavioralCloningTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.reset_history()

    def reset_history(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.per_action_accuracies = []
    
    def save_checkpoint(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    
    def load_checkpoint(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        val_loss, exact_matches, total_samples = 0, 0, 0
        action_correct = torch.zeros(5, device=self.device)
        action_total = torch.zeros(5, device=self.device)
        
        with torch.no_grad():
            for images, other_features, actions in val_loader:
                images, other_features, actions = images.to(self.device), other_features.to(self.device), actions.to(self.device)
                outputs = self.model(images, other_features)
                
                loss = criterion(outputs, actions)
                val_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                exact_matches += (predicted == actions).all(dim=1).sum().item()
                total_samples += actions.size(0)
                
                for i in range(5):
                    action_correct[i] += (predicted[:, i] == actions[:, i]).sum().item()
                    action_total[i] += actions.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        exact_match_accuracy = exact_matches / total_samples
        per_action_acc = [corr.item() / total.item() if total.item() > 0 else 0 for corr, total in zip(action_correct, action_total)]
        
        return avg_val_loss, exact_match_accuracy, per_action_acc
    
    def plot_training_history(self, save_path='training_history.png'):
        if not self.train_losses: return
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        action_names = ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']
        
        axes[0, 0].plot(self.train_losses, label='Training Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        axes[0, 1].plot(self.val_accuracies, label='Exact Match Accuracy', color='green')
        axes[0, 1].set_title('Exact Match Accuracy')
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        per_action_array = np.array(self.per_action_accuracies)
        for i, action in enumerate(action_names):
            axes[1, 0].plot(per_action_array[:, i], label=action)
        axes[1, 0].set_title('Per-Action Accuracy')
        axes[1, 0].grid(True)
        axes[1, 0].legend()

        final_accuracies = self.per_action_accuracies[-1]
        bars = axes[1, 1].bar(action_names, final_accuracies)
        axes[1, 1].set_title('Final Per-Action Accuracy')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)
        for bar, acc in zip(bars, final_accuracies):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Training history plot saved as '{save_path}'")

    def train(self, train_loader, val_loader, epochs, learning_rate, weight_decay, loss_type):
        self.reset_history()
        
        pos_weights = []
        for i, col in enumerate(train_loader.dataset.action_columns):
            data_source = train_loader.dataset.data
            pos = data_source[col].sum()
            neg = len(data_source) - pos
            pos_weights.append(neg / pos if pos > 0 else 1.0)
        
        criterion = FocalLoss(pos_weight=torch.FloatTensor(pos_weights).to(self.device)) if loss_type == 'focal' else nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(pos_weights).to(self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        print(f"\nTraining on {self.device} with {loss_type.upper()} loss...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        patience_counter, patience = 0, 15

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for images, other_features, actions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, other_features, actions = images.to(self.device), other_features.to(self.device), actions.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images, other_features)
                loss = criterion(outputs, actions)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            val_loss, val_accuracy, per_action_acc = self.validate(val_loader, criterion)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            self.per_action_accuracies.append(per_action_acc)
            
            per_action_str = ", ".join([f"{acc:.2f}" for acc in per_action_acc])
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Exact Match: {val_accuracy:.3f}, Actions: [{per_action_str}]")
            
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs.")
                break
        
        self.load_checkpoint('best_model.pth')
        print("Training completed!")
        return best_val_loss, self.per_action_accuracies[-1] if self.per_action_accuracies else [0]*5

def export_model_for_inference(model, image_size, output_dir='model'):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    
    model_info = {
        'model_class': model.__class__.__name__,
        'image_shape': [1, image_size[1], image_size[0]], # C, H, W
        'other_features_dim': 4,
        'num_actions': 5,
        'action_columns': ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']
    }
    with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model exported to {output_dir}/")

def load_model_for_inference(model_dir='model'):
    """
    Loads a CNN model and its metadata for inference.
    """
    model_info_path = os.path.join(model_dir, 'model_info.json')
    if not os.path.exists(model_info_path):
        raise FileNotFoundError(f"model_info.json not found in {model_dir}")

    with open(model_info_path, 'r') as f:
        model_info = json.load(f)

    # Recreate model from the info file
    model = BehavioralCloningNet(
        image_shape=tuple(model_info['image_shape']),
        other_features_dim=model_info['other_features_dim'],
        num_actions=model_info['num_actions']
    )
    
    # Load weights
    weights_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model, model_info

def main():
    """Main script to train an ensemble of quality-controlled CNN models."""
    # --- Configuration ---
    DATA_DIR = r"C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\HKData"
    TRAIN_ON_ALL_DATA = True # Set to False to train on only the latest data session
    ENABLE_DATA_AUGMENTATION = False # Set to False to disable image augmentations
    NUM_MODELS_TO_FIND = 3
    MAX_TRIALS = 100
    MIN_ACTION_ACCURACY = 0.5

    # --- Data Loading ---
    csv_files = [f for f in os.listdir(DATA_DIR) if f.startswith('hk_actions_') and f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}. Exiting."); return

    data_to_load = []
    if TRAIN_ON_ALL_DATA:
        print("Loading all data sessions...")
        data_to_load = csv_files
    else:
        print("Loading latest data session...")
        data_to_load.append(max(csv_files))

    all_dfs = []
    for csv_file in data_to_load:
        session_id = csv_file.replace('hk_actions_', '').replace('.csv', '')
        frames_dir = os.path.join(DATA_DIR, f'frames_{session_id}')
        if os.path.exists(frames_dir):
            df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
            df['frame_path'] = df['frame_id'].apply(lambda x: os.path.join(frames_dir, f"frame_{x:06d}.png"))
            all_dfs.append(df)
    
    if not all_dfs: print("No data could be loaded. Exiting."); return
    master_df = pd.concat(all_dfs, ignore_index=True)
    if len(master_df) == 0: print("No valid data in master DataFrame. Exiting."); return


    # Data Augmentation (for training only)
    if ENABLE_DATA_AUGMENTATION:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ToTensor() # Convert PIL Image to PyTorch Tensor
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    # No augmentation for validation, just ToTensor
    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # --- Ensemble Training via Filter and Collect ---
    print(f"\n--- Searching for {NUM_MODELS_TO_FIND} 'Good' Models (Min Action Accuracy > {MIN_ACTION_ACCURACY}) ---")
    print(f"Maximum trials: {MAX_TRIALS}\n")
    
    search_space = {
        'learning_rate': [5e-5, 1e-4, 5e-4],
        'dropout_rate': [0.3, 0.4, 0.5],
        'weight_decay': [1e-5, 1e-4, 5e-4],
        'loss_type': ['bce', 'focal'],
        'image_size': [(80,40),(160, 75),(320,130)],
        'batch_size': [32, 64]
    }
    
    saved_models_count = 0
    for trial in range(MAX_TRIALS):
        if saved_models_count >= NUM_MODELS_TO_FIND:
            print(f"Successfully found {NUM_MODELS_TO_FIND} models. Halting search.")
            break

        print(f"\n--- Trial {trial + 1}/{MAX_TRIALS} | Models Found: {saved_models_count}/{NUM_MODELS_TO_FIND} ---")
        
        hp = {k: random.choice(v) for k, v in search_space.items()}
        print(f"Sampled Hyperparameters: {json.dumps(hp, indent=2)}")
        
        # Split dataframes for this trial
        train_df, val_df = train_test_split(master_df, test_size=0.2, random_state=42) # Using a fixed random_state for reproducibility

        # Create datasets for this trial with specific image_size and transforms
        train_trial_dataset = HollowKnightDataset(
            data_df=train_df,
            image_size=hp['image_size'],
            transform=train_transform
        )
        val_trial_dataset = HollowKnightDataset(
            data_df=val_df,
            image_size=hp['image_size'],
            transform=val_transform
        )
        
        if len(train_trial_dataset) == 0 or len(val_trial_dataset) == 0:
            print(f"Skipping trial {trial+1}: Not enough valid data found after splitting for image size {hp['image_size']}")
            continue
            
        train_loader = DataLoader(train_trial_dataset, batch_size=hp['batch_size'], shuffle=True)
        val_loader = DataLoader(val_trial_dataset, batch_size=hp['batch_size'], shuffle=False)
        
        model = BehavioralCloningNet(
            image_shape=(1, hp['image_size'][1], hp['image_size'][0]), # C, H, W
            dropout_rate=hp['dropout_rate']
        )
        
        trainer = BehavioralCloningTrainer(model)
        _, final_accuracies = trainer.train(
            train_loader, val_loader, epochs=100,
            learning_rate=hp['learning_rate'], 
            weight_decay=hp['weight_decay'],
            loss_type=hp['loss_type']
        )
        
        # --- Quality Control Check ---
        if all(acc > MIN_ACTION_ACCURACY for acc in final_accuracies):
            saved_models_count += 1
            print(f"\nSUCCESS! Model passed quality check. Saving as ensemble member #{saved_models_count}.")
            output_dir = f"model/ensemble_{saved_models_count}"
            export_model_for_inference(model, hp['image_size'], output_dir=output_dir)
            trainer.plot_training_history(save_path=os.path.join(output_dir, 'training_history.png'))
            # Save hyperparameters with the model
            with open(os.path.join(output_dir, 'hyperparameters.json'), 'w') as f:
                json.dump(hp, f, indent=2)
        else:
            print("\nFAILURE. Model did not meet the minimum accuracy for all actions. Discarding.")

    print(f"\n--- Search Complete ---")
    print(f"Found {saved_models_count} models that met the quality criteria.")

    # --- Cleanup ---
    if os.path.exists('best_model.pth'):
        os.remove('best_model.pth')
        print("Cleaned up temporary checkpoint file.")

if __name__ == "__main__":
    import torch.nn.functional as F
    main()
