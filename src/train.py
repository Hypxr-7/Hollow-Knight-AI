# Standard library imports
import os
import json
import random
import warnings
import datetime

# Third-party imports
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp

# Warnings configuration
warnings.filterwarnings('ignore')

def get_transforms(img_size, is_train=False):
    """
    Returns the transformation pipeline.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # Geometric jitter (scaling) without rotation/translation
            transforms.RandomAffine(degrees=0, translate=None, scale=(0.95, 1.05)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
        ])

def engineer_features(df):
    """
    Adds derived features to the dataframe.
    Assumes df contains a single contiguous session or is carefully handled.
    """
    # 1. Deltas (Velocity)
    df['player_dx'] = df['x_position'].diff().fillna(0)
    df['player_dy'] = df['y_position'].diff().fillna(0)
    df['enemy_dx'] = df['enemy_x'].diff().fillna(0)
    df['enemy_dy'] = df['enemy_y'].diff().fillna(0)
    
    # 2. Speed (Magnitude)
    df['player_speed'] = np.sqrt(df['player_dx']**2 + df['player_dy']**2)
    
    # 3. Euclidean Distance to Enemy
    df['enemy_distance'] = np.sqrt((df['x_position'] - df['enemy_x'])**2 + (df['y_position'] - df['enemy_y'])**2)
    
    # 4. Relative Position (Enemy relative to Player)
    df['rel_x'] = df['enemy_x'] - df['x_position']
    df['rel_y'] = df['enemy_y'] - df['y_position']

    # 5. Relative Velocity (Difference in velocities)
    df['vel_diff_x'] = df['player_dx'] - df['enemy_dx']
    df['vel_diff_y'] = df['player_dy'] - df['enemy_dy']
    
    # 6. Temporal Stacking (Locality for 3 frames)
    # We create history for scalar features so that shuffling rows doesn't break the context
    cols_to_shift = [
        'x_position', 'y_position', 'health', 'enemy_x', 'enemy_y',
        'player_dx', 'player_dy', 'enemy_dx', 'enemy_dy',
        'rel_x', 'rel_y', 'enemy_distance'
    ]
    action_cols = ['moving_left', 'moving_right', 'moving_up', 'moving_down', 'attacking', 'jumping', 'dashing', 'casting']
    
    # Backward compatibility: Ensure new columns exist
    for col in ['moving_up', 'moving_down', 'casting']:
        if col not in df.columns:
            df[col] = 0
            
    if 'health' not in df.columns:
        df['health'] = 5 # Default assumption for old data

    for col in cols_to_shift + action_cols:
        df[f'{col}_t1'] = df[col].shift(1).fillna(0)
        df[f'{col}_t2'] = df[col].shift(2).fillna(0)
        
    return df

class HollowKnightDataset(Dataset):
    """
    Dataset for Hollow Knight behavioral cloning.
    Prepares image and positional data for the CNN model.
    Supports frame stacking and sequence generation for LSTM.
    """
    def __init__(self, data_df, transform=None, image_size=(80, 60), n_frames=1, sequence_length=1, feature_columns=None):
        self.data = data_df.copy().reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size
        self.n_frames = n_frames
        self.sequence_length = sequence_length
        self.action_columns = ['moving_left', 'moving_right', 'moving_up', 'moving_down', 'attacking', 'jumping', 'dashing', 'casting']
        
        if feature_columns is None:
            self.feature_columns = ['x_position', 'y_position', 'enemy_x', 'enemy_y']
        else:
            self.feature_columns = feature_columns

        # Verify features exist
        missing_cols = [c for c in self.feature_columns if c not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in dataframe: {missing_cols}")
            
        if not self.image_size:
            self.image_size = self._get_original_image_size()

        # Pre-compute valid indices for sequences
        self.valid_indices = []
        if self.sequence_length > 1:
            if 'session_id' not in self.data.columns:
                 # Fallback if session_id is missing (treat as one session)
                 self.valid_indices = list(range(len(self.data) - self.sequence_length + 1))
            else:
                # We need valid sequences where all frames belong to the same session
                for i in range(len(self.data) - self.sequence_length + 1):
                    if self.data.iloc[i]['session_id'] == self.data.iloc[i + self.sequence_length - 1]['session_id']:
                        self.valid_indices.append(i)
        else:
            self.valid_indices = list(range(len(self.data)))

    def _get_original_image_size(self):
        if len(self.data) > 0:
            first_frame_path = self.data['frame_path'].iloc[0]
            if os.path.exists(first_frame_path):
                with Image.open(first_frame_path) as img:
                    return img.size 
        return None

    def _get_previous_frame_path(self, current_path, current_id, offset):
        dir_name = os.path.dirname(current_path)
        prev_id = current_id - offset
        return os.path.join(dir_name, f"frame_{prev_id:06d}.png")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        # Output shapes: (Seq, Stack*C, H, W), (Seq, Feats), (Seq, Actions)
        seq_images = []
        seq_features = []
        seq_actions = []

        range_end = start_idx + self.sequence_length
        
        for i in range(start_idx, range_end):
            row = self.data.iloc[i]
            current_frame_id = row['frame_id']
            current_frame_path = row['frame_path']
            
            frames = []
            for j in range(self.n_frames):
                if j == 0:
                    path = current_frame_path
                else:
                    path = self._get_previous_frame_path(current_frame_path, current_frame_id, j)
                
                if os.path.exists(path):
                    with Image.open(path).convert('L') as img:
                        if self.image_size and not self.transform:
                            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
                        
                        if self.transform:
                            img_tensor = self.transform(img)
                            if not isinstance(img_tensor, torch.Tensor):
                                 img_tensor = transforms.ToTensor()(img_tensor)
                        else:
                            img_tensor = transforms.ToTensor()(img)
                        frames.append(img_tensor)
                else:
                    if frames:
                        frames.append(frames[-1])
                    else:
                        h, w = self.image_size[1], self.image_size[0]
                        frames.append(torch.zeros(1, h, w))

            image_stack = torch.cat(frames, dim=0)
            other_features = torch.FloatTensor([float(row[col]) for col in self.feature_columns])
            actions = torch.FloatTensor([float(row[col]) for col in self.action_columns])
            
            seq_images.append(image_stack)
            seq_features.append(other_features)
            seq_actions.append(actions)
            
        return torch.stack(seq_images), torch.stack(seq_features), torch.stack(seq_actions)


class ResNetLSTMBehavioralCloningNet(nn.Module):
    """
    ResNet + LSTM architecture for temporal behavioral cloning.
    Processes a sequence of frames and outputs a sequence of actions.
    """
    def __init__(self, image_shape=(3, 160, 320), other_features_dim=4, num_actions=8, hidden_dim=256, num_layers=1, dropout_rate=0.5):
        super(ResNetLSTMBehavioralCloningNet, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder (ResNet18)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        original_first_conv = self.resnet.conv1
        if image_shape[0] != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels=image_shape[0], 
                out_channels=original_first_conv.out_channels,
                kernel_size=original_first_conv.kernel_size,
                stride=original_first_conv.stride,
                padding=original_first_conv.padding,
                bias=original_first_conv.bias is not None
            )
            nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        self.num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Fusion
        self.fusion_dim = 256
        self.fusion_fc = nn.Linear(self.num_ftrs + other_features_dim, self.fusion_dim)
        self.bn1 = nn.BatchNorm1d(self.fusion_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.fusion_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Head
        self.final_fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, images, other_features, hidden=None):
        """
        images: (Batch, Seq, Channels, H, W)
        other_features: (Batch, Seq, FeatDim)
        """
        batch_size, seq_len, C, H, W = images.size()
        
        # 1. Flatten Batch and Sequence for CNN
        # (B*S, C, H, W)
        cnn_in = images.view(batch_size * seq_len, C, H, W)
        feat_in = other_features.view(batch_size * seq_len, -1)
        
        # 2. Extract Features
        cnn_out = self.resnet(cnn_in)
        
        # 3. Fuse
        combined = torch.cat([cnn_out, feat_in], dim=1)
        fused = F.relu(self.bn1(self.fusion_fc(combined)))
        fused = self.dropout(fused)
        
        # 4. Reshape for LSTM
        # (B, S, FusionDim)
        lstm_in = fused.view(batch_size, seq_len, -1)
        
        # 5. LSTM
        lstm_out, hidden = self.lstm(lstm_in, hidden)
        
        # 6. Final FC
        # (B, S, Hidden) -> (B*S, Hidden) -> FC -> (B, S, Actions)
        fc_in = lstm_out.reshape(batch_size * seq_len, -1)
        out = self.final_fc(fc_in)
        out = out.view(batch_size, seq_len, -1)
        
        return out, hidden

class ResNetBehavioralCloningNet(nn.Module):
    """
    ResNet-based CNN for multi-label behavioral cloning.
    Uses a ResNet18 backbone for better feature extraction.
    """
    def __init__(self, image_shape=(3, 160, 320), other_features_dim=4, num_actions=8, dropout_rate=0.5):
        super(ResNetBehavioralCloningNet, self).__init__()
        
        # Load Pretrained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify first layer to accept 'image_shape[0]' channels (frame stack depth)
        original_first_conv = self.resnet.conv1
        if image_shape[0] != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels=image_shape[0], 
                out_channels=original_first_conv.out_channels,
                kernel_size=original_first_conv.kernel_size,
                stride=original_first_conv.stride,
                padding=original_first_conv.padding,
                bias=original_first_conv.bias is not None
            )
            nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # Remove the fully connected layer
        self.num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Fusion Layer
        self.fusion_fc = nn.Linear(self.num_ftrs + other_features_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout_rate)
        self.final_fc = nn.Linear(256, num_actions)

    def forward(self, image, other_features):
        x = self.resnet(image)
        combined = torch.cat([x, other_features], dim=1)
        x = F.relu(self.bn1(self.fusion_fc(combined)))
        x = self.dropout(x)
        output = self.final_fc(x)
        return output

# Alias for backward compatibility
BehavioralCloningNet = ResNetBehavioralCloningNet


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
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', log_dir=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None
        self.scaler = amp.GradScaler() # For Mixed Precision
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
        total_correct_bits = 0
        action_correct = torch.zeros(8, device=self.device)
        action_total = torch.zeros(8, device=self.device)
        
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, other_features, actions in val_loader:
                images, other_features, actions = images.to(self.device), other_features.to(self.device), actions.to(self.device)
                
                with amp.autocast(enabled=True):
                    # Handle LSTM tuple return
                    if isinstance(self.model, ResNetLSTMBehavioralCloningNet):
                        outputs, _ = self.model(images, other_features)
                    else:
                        outputs = self.model(images, other_features)
                    
                    # Flatten sequence dimension for loss
                    if outputs.dim() == 3:
                        outputs = outputs.reshape(-1, outputs.size(-1))
                        actions = actions.reshape(-1, actions.size(-1))
                        
                    loss = criterion(outputs, actions)
                
                val_loss += loss.item()
                
                # Store for threshold finding
                probs = torch.sigmoid(outputs)
                all_labels.append(actions.cpu())
                all_preds.append(probs.cpu())

                predicted = (probs > 0.5).float() # Default 0.5 for logging progress
                exact_matches += (predicted == actions).all(dim=1).sum().item()
                total_correct_bits += (predicted == actions).sum().item()
                total_samples += actions.size(0)
                
                for i in range(8):
                    action_correct[i] += (predicted[:, i] == actions[:, i]).sum().item()
                    action_total[i] += actions.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        exact_match_accuracy = exact_matches / total_samples
        partial_match_accuracy = total_correct_bits / (total_samples * 8)
        per_action_acc = [corr.item() / total.item() if total.item() > 0 else 0 for corr, total in zip(action_correct, action_total)]
        
        # Concatenate for global metrics
        if len(all_labels) > 0:
            all_labels = torch.cat(all_labels).numpy()
            all_preds = torch.cat(all_preds).numpy()
        else:
            all_labels = np.array([])
            all_preds = np.array([])
        
        return avg_val_loss, exact_match_accuracy, partial_match_accuracy, per_action_acc, all_labels, all_preds
    
    def find_optimal_thresholds(self, labels, preds, action_names):
        """Finds the best threshold for each action to maximize F1 Score."""
        thresholds = {}
        if len(labels) == 0: return {a: 0.5 for a in action_names}

        print("\n--- Optimal Threshold Search (Max F1 Score) ---")
        
        for i, action in enumerate(action_names):
            precision, recall, thresh = precision_recall_curve(labels[:, i], preds[:, i])
            
            # Calculate F1 for each threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            
            # precision_recall_curve returns thresholds with length = len(precision) - 1
            if best_idx < len(thresh):
                best_threshold = thresh[best_idx]
            else:
                best_threshold = 0.5 # Default fallback
            
            thresholds[action] = float(best_threshold)
            print(f"{action}: {best_threshold:.4f} (F1: {f1_scores[best_idx]:.4f})")
            
        return thresholds

    def plot_training_history(self, save_path='training_history.png'):
        if not self.train_losses: return
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        action_names = ['moving_left', 'moving_right', 'moving_up', 'moving_down', 'attacking', 'jumping', 'dashing', 'casting']
        
        axes[0, 0].plot(self.train_losses, label='Training Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        axes[0, 1].plot(self.val_accuracies, label='Exact Match Accuracy', color='green')
        axes[0, 1].plot(self.partial_accuracies, label='Partial Match Accuracy', color='orange', linestyle='--')
        axes[0, 1].set_title('Accuracy')
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

    def train(self, train_loader, val_loader, epochs, learning_rate, weight_decay, loss_type):
        self.reset_history()
        self.partial_accuracies = []
        
        pos_weights = []
        for i, col in enumerate(train_loader.dataset.action_columns):
            data_source = train_loader.dataset.data
            pos = data_source[col].sum()
            neg = len(data_source) - pos
            pos_weights.append(neg / pos if pos > 0 else 1.0)
        
        criterion = FocalLoss(pos_weight=torch.FloatTensor(pos_weights).to(self.device)) if loss_type == 'focal' else nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(pos_weights).to(self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        print(f"\nTraining on {self.device} with {loss_type.upper()} loss (Mixed Precision Enabled)...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        patience_counter, patience = 0, 15
        
        global_step = 0
        
        best_labels = None
        best_preds = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, other_features, actions in pbar:
                images, other_features, actions = images.to(self.device), other_features.to(self.device), actions.to(self.device)
                optimizer.zero_grad()
                
                with amp.autocast(enabled=True):
                    # Handle LSTM tuple return
                    if isinstance(self.model, ResNetLSTMBehavioralCloningNet):
                        outputs, _ = self.model(images, other_features)
                    else:
                        outputs = self.model(images, other_features)
                    
                    # Flatten sequence dimension for loss
                    if outputs.dim() == 3:
                        outputs = outputs.view(-1, outputs.size(-1))
                        actions = actions.view(-1, actions.size(-1))
                        
                    loss = criterion(outputs, actions)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                
                train_loss += loss.item()
                global_step += 1
                
                if self.writer:
                    self.writer.add_scalar('Loss/batch_train', loss.item(), global_step)

            avg_train_loss = train_loss / len(train_loader)
            val_loss, val_accuracy, partial_accuracy, per_action_acc, val_labels, val_preds = self.validate(val_loader, criterion)
            scheduler.step(val_loss)
            
            if self.writer:
                self.writer.add_scalar('Loss/epoch_train', avg_train_loss, epoch)
                self.writer.add_scalar('Loss/epoch_val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/exact_match', val_accuracy, epoch)
                self.writer.add_scalar('Accuracy/partial_match', partial_accuracy, epoch)
                for i, name in enumerate(['Left', 'Right', 'Attack', 'Jump', 'Dash']):
                    self.writer.add_scalar(f'Accuracy_Per_Action/{name}', per_action_acc[i], epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pth')
                best_labels = val_labels
                best_preds = val_preds
            else:
                patience_counter += 1
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            self.partial_accuracies.append(partial_accuracy)
            self.per_action_accuracies.append(per_action_acc)
            
            per_action_str = ", ".join([f"{acc:.2f}" for acc in per_action_acc])
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Exact: {val_accuracy:.3f} | Partial: {partial_accuracy:.3f} | Actions: [{per_action_str}]")
            
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs.")
                break
        
        if self.writer:
            self.writer.close()
            
        self.load_checkpoint('best_model.pth')
        print("Training completed!")
        
        optimal_thresholds = self.find_optimal_thresholds(best_labels, best_preds, train_loader.dataset.action_columns)
        
        return best_val_loss, self.per_action_accuracies[-1] if self.per_action_accuracies else [0]*5, optimal_thresholds

def export_model_for_inference(model, image_size, n_frames, feature_columns, thresholds, scaler=None, output_dir='model'):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    
    model_info = {
        'model_class': model.__class__.__name__,
        'image_shape': [n_frames, image_size[1], image_size[0]],
        'other_features_dim': len(feature_columns),
        'feature_columns': feature_columns,
        'num_actions': 8,
        'action_columns': ['moving_left', 'moving_right', 'moving_up', 'moving_down', 'attacking', 'jumping', 'dashing', 'casting'],
        'n_frames': n_frames,
        'sequence_length': getattr(model, 'sequence_length', 1),
        'hidden_dim': getattr(model, 'hidden_dim', 256),
        'thresholds': thresholds
    }
    
    if scaler:
        model_info['scaler_mean'] = scaler.mean_.tolist()
        model_info['scaler_scale'] = scaler.scale_.tolist()

    with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model exported to {output_dir}/")

def load_model_for_inference(model_dir='model'):
    model_info_path = os.path.join(model_dir, 'model_info.json')
    if not os.path.exists(model_info_path):
        raise FileNotFoundError(f"model_info.json not found in {model_dir}")

    with open(model_info_path, 'r') as f:
        model_info = json.load(f)

    n_frames = model_info.get('n_frames', 1)
    
    feature_columns = model_info.get('feature_columns', ['x_position', 'y_position', 'enemy_x', 'enemy_y'])
    other_features_dim = model_info.get('other_features_dim', 4)

    # Reconstruct the Scaler if info is present
    scaler = None
    if 'scaler_mean' in model_info and 'scaler_scale' in model_info:
        scaler = StandardScaler()
        scaler.mean_ = np.array(model_info['scaler_mean'])
        scaler.scale_ = np.array(model_info['scaler_scale'])
        # Scikit-learn requires these to be set for the scaler to be considered "fitted"
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_samples_seen_ = 1000 # Dummy value

    model_class = model_info.get('model_class', 'ResNetBehavioralCloningNet')
    
    if model_class == 'ResNetLSTMBehavioralCloningNet':
        model = ResNetLSTMBehavioralCloningNet(
            image_shape=tuple(model_info['image_shape']),
            other_features_dim=other_features_dim,
            num_actions=model_info['num_actions'],
            hidden_dim=model_info.get('hidden_dim', 256)
        )
    else:
        model = ResNetBehavioralCloningNet(
            image_shape=tuple(model_info['image_shape']),
            other_features_dim=other_features_dim,
            num_actions=model_info['num_actions']
        )
    
    weights_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    
    model_info['feature_columns'] = feature_columns
    
    return model, model_info, scaler

def main():
    """Main script to train an ensemble of quality-controlled CNN models."""
    # --- Configuration ---
    DATA_DIR = r"C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\HKData"
    TRAIN_ON_ALL_DATA = True
    NUM_MODELS_TO_FIND = 3
    MAX_TRIALS = 100
    MIN_ACTION_ACCURACY = 0.5
    STACK_SIZE = 3 
    SEQUENCE_LENGTH = 8 
    
    BASE_LOG_DIR = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(BASE_LOG_DIR, exist_ok=True)

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
    print("Loading and Preprocessing Data...")
    for csv_file in tqdm(data_to_load, desc="Processing Sessions"):
        session_id = csv_file.replace('hk_actions_', '').replace('.csv', '')
        frames_dir = os.path.join(DATA_DIR, f'frames_{session_id}')
        if os.path.exists(frames_dir):
            df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
            df['frame_path'] = df['frame_id'].apply(lambda x: os.path.join(frames_dir, f"frame_{x:06d}.png"))
            
            # Add session_id
            df['session_id'] = session_id
            
            # Filter missing files upfront to prevent Sampler misalignment
            valid_mask = df['frame_path'].apply(os.path.exists)
            df = df[valid_mask].copy()

            df = engineer_features(df)
            all_dfs.append(df)
    
    if not all_dfs: print("No data could be loaded. Exiting."); return
    master_df = pd.concat(all_dfs, ignore_index=True)
    if len(master_df) == 0: print("No valid data in master DataFrame. Exiting."); return

    # Define the full set of 19 + (16*2) features = 51 features
    base_features = [
        'x_position', 'y_position', 'health', 'enemy_x', 'enemy_y', 
        'player_dx', 'player_dy', 'enemy_dx', 'enemy_dy', 
        'player_speed', 'enemy_distance', 
        'rel_x', 'rel_y', 'vel_diff_x', 'vel_diff_y',
        'moving_left', 'moving_right', 'moving_up', 'moving_down', 'attacking', 'jumping', 'dashing', 'casting'
    ]
    
    feature_columns = base_features.copy()
    for col in [
        'x_position', 'y_position', 'health', 'enemy_x', 'enemy_y',
        'player_dx', 'player_dy', 'enemy_dx', 'enemy_dy',
        'rel_x', 'rel_y', 'enemy_distance',
        'moving_left', 'moving_right', 'moving_up', 'moving_down', 'attacking', 'jumping', 'dashing', 'casting'
    ]:
        feature_columns.append(f'{col}_t1')
        feature_columns.append(f'{col}_t2')
    
    # Filter out columns that aren't inputs (labels are target, but here we use history as input)
    input_feature_columns = [c for c in feature_columns if c not in ['moving_left', 'moving_right', 'moving_up', 'moving_down', 'attacking', 'jumping', 'dashing', 'casting']]
    
    print(f"Using {len(input_feature_columns)} input features.")

    # --- Feature Normalization ---
    print("Fitting StandardScaler on scalar features...")
    scaler = StandardScaler()
    master_df[input_feature_columns] = scaler.fit_transform(master_df[input_feature_columns])

    print(f"\n--- Searching for {NUM_MODELS_TO_FIND} 'Good' Models (Min Action Accuracy > {MIN_ACTION_ACCURACY}) ---")
    print(f"Frame Stacking: {STACK_SIZE}")
    print(f"Sequence Length: {SEQUENCE_LENGTH}")
    print(f"TensorBoard Logs: {BASE_LOG_DIR}")
    
    search_space = {
        'learning_rate': [1e-4, 2e-4],
        'dropout_rate': [0.4, 0.5],
        'weight_decay': [1e-4, 5e-4],
        'loss_type': ['focal'],
        'image_size': [(224, 224)], # ResNet standard size works well
        'batch_size': [16, 24], 
        'hidden_dim': [256, 512]
    }
    
    saved_models_count = 0
    for trial in range(MAX_TRIALS):
        if saved_models_count >= NUM_MODELS_TO_FIND:
            print(f"Successfully found {NUM_MODELS_TO_FIND} models. Halting search.")
            break

        print(f"\n--- Trial {trial + 1}/{MAX_TRIALS} | Models Found: {saved_models_count}/{NUM_MODELS_TO_FIND} ---")
        
        hp = {k: random.choice(v) for k, v in search_space.items()}
        print(f"Sampled Hyperparameters: {json.dumps(hp, indent=2)}")
        
        train_df, val_df = train_test_split(master_df, test_size=0.2, random_state=42)
        
        # Reset Index strictly for the sampler to align with the Dataset
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        # Sequence-based training means simple WeightedRandomSampler is hard.
        # We disable it for now and rely on shuffling + Focal Loss.
        
        train_trial_dataset = HollowKnightDataset(
            data_df=train_df,
            image_size=hp['image_size'],
            transform=get_transforms(hp['image_size'], is_train=True),
            n_frames=STACK_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            feature_columns=input_feature_columns
        )
        val_trial_dataset = HollowKnightDataset(
            data_df=val_df,
            image_size=hp['image_size'],
            transform=get_transforms(hp['image_size'], is_train=False), 
            n_frames=STACK_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            feature_columns=input_feature_columns
        )
        
        if len(train_trial_dataset) == 0:
            print(f"Skipping trial {trial+1}: Not enough data.")
            continue
            
        train_loader = DataLoader(
            train_trial_dataset, 
            batch_size=hp['batch_size'], 
            shuffle=True, 
            num_workers=0
        )
        val_loader = DataLoader(
            val_trial_dataset, 
            batch_size=hp['batch_size'], 
            shuffle=False
        )
        
        model = ResNetLSTMBehavioralCloningNet(
            image_shape=(STACK_SIZE, hp['image_size'][1], hp['image_size'][0]), 
            other_features_dim=len(input_feature_columns),
            dropout_rate=hp['dropout_rate'],
            hidden_dim=hp['hidden_dim']
        )
        model.sequence_length = SEQUENCE_LENGTH
        
        trial_log_dir = os.path.join(BASE_LOG_DIR, f"trial_{trial+1}")
        
        trainer = BehavioralCloningTrainer(model, log_dir=trial_log_dir)
        _, final_accuracies, optimal_thresholds = trainer.train(
            train_loader, val_loader, epochs=100,
            learning_rate=hp['learning_rate'], 
            weight_decay=hp['weight_decay'],
            loss_type=hp['loss_type']
        )
        
        if all(acc > MIN_ACTION_ACCURACY for acc in final_accuracies):
            saved_models_count += 1
            print(f"\nSUCCESS! Model passed quality check. Saving as ensemble member #{saved_models_count}.")
            output_dir = f"model/ensemble_{saved_models_count}"
            export_model_for_inference(
                model, hp['image_size'], STACK_SIZE, input_feature_columns, 
                optimal_thresholds, scaler=scaler, output_dir=output_dir
            )
            trainer.plot_training_history(save_path=os.path.join(output_dir, 'training_history.png'))
            with open(os.path.join(output_dir, 'hyperparameters.json'), 'w') as f:
                json.dump(hp, f, indent=2)
        else:
            print("\nFAILURE. Model did not meet the minimum accuracy for all actions. Discarding.")

    print(f"\n--- Search Complete ---")
    print(f"Found {saved_models_count} models that met the quality criteria.")

    if os.path.exists('best_model.pth'):
        os.remove('best_model.pth')

if __name__ == "__main__":
    main()
