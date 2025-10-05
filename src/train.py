# Standard library imports
import os
import json
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

# Warnings configuration
warnings.filterwarnings('ignore')

class HollowKnightDataset(Dataset):
    def __init__(self, csv_path, frames_path, transform=None, image_size=(80, 60)):
        """
        Dataset for Hollow Knight behavioral cloning
        
        Args:
            csv_path: Path to CSV file with actions
            frames_path: Path to frames directory
            transform: Image transformations
            image_size: Size to resize images to (width, height)
        """
        self.data = pd.read_csv(csv_path)
        self.frames_path = frames_path
        self.transform = transform
        self.image_size = image_size
        
        # Updated action columns to match new CSV structure
        self.action_columns = ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']
        
        # Filter data to only include frames with existing images
        self.valid_indices = self._get_valid_indices()
        self.data = self.data.iloc[self.valid_indices].reset_index(drop=True)
        
        print(f"Dataset loaded: {len(self.data)} samples with valid frames")
        print(f"Image size: {self.image_size} (flattened: {self.image_size[0] * self.image_size[1]} features)")
        
        # Analyze action distribution
        print("\n=== Action Distribution ===")
        for col in self.action_columns:
            count = self.data[col].sum()
            pct = (count / len(self.data)) * 100
            print(f"{col:15s}: {int(count):5d} frames ({pct:.1f}%)")
        
        # Analyze enemy presence
        enemy_present = ((self.data['enemy_x'] != 0) | (self.data['enemy_y'] != 0)).sum()
        enemy_pct = (enemy_present / len(self.data)) * 100
        print(f"{'enemy_present':15s}: {int(enemy_present):5d} frames ({enemy_pct:.1f}%)")
    
    def _get_valid_indices(self):
        """Get indices of frames that have corresponding images"""
        valid_indices = []
        for idx, row in self.data.iterrows():
            frame_path = os.path.join(self.frames_path, f"frame_{row['frame_id']:06d}.png")
            if os.path.exists(frame_path):
                valid_indices.append(idx)
        return valid_indices
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        frame_path = os.path.join(self.frames_path, f"frame_{row['frame_id']:06d}.png")
        image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize image
        image = cv2.resize(image, self.image_size)
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        if self.transform:
            image = self.transform(image)
        
        # Flatten image for neural network
        image_features = torch.FloatTensor(image).flatten()
        
        # Add position and enemy information as additional features
        player_x = torch.FloatTensor([row['x_position'] / 1920.0])  # Normalize assuming 1920 screen width
        player_y = torch.FloatTensor([row['y_position'] / 1080.0])  # Normalize assuming 1080 screen height
        enemy_x = torch.FloatTensor([row['enemy_x'] / 1920.0])
        enemy_y = torch.FloatTensor([row['enemy_y'] / 1080.0])
        
        # Combine image features with position data
        features = torch.cat([image_features, player_x, player_y, enemy_x, enemy_y])
        
        # Get actions as binary vector
        actions = torch.FloatTensor([
            float(row[col]) for col in self.action_columns
        ])
        
        return features, actions


class BehavioralCloningNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], output_dim=5, dropout_rate=0.3):
        """
        Neural network for multi-label behavioral cloning
        
        Args:
            input_dim: Input dimension (image features + position data)
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension (number of actions)
            dropout_rate: Dropout rate for regularization
        """
        super(BehavioralCloningNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer - NO SIGMOID (BCEWithLogitsLoss handles it)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)




class BehavioralCloningTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.per_action_accuracies = []
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        torch.save(self.model.state_dict(), filepath)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
    
    def validate(self, val_loader, criterion):
        """Validate the model with multi-label metrics"""
        self.model.eval()
        val_loss = 0
        exact_matches = 0
        total_samples = 0
        
        # Per-action accuracy tracking
        action_correct = torch.zeros(5).to(self.device)
        action_total = torch.zeros(5).to(self.device)
        
        with torch.no_grad():
            for features, actions in val_loader:
                features, actions = features.to(self.device), actions.to(self.device)
                outputs = self.model(features)
                
                # Calculate loss
                loss = criterion(outputs, actions)
                val_loss += loss.item()
                
                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                
                # Exact match accuracy (all actions must match)
                exact_match = (predicted == actions).all(dim=1).float()
                exact_matches += exact_match.sum().item()
                total_samples += actions.size(0)
                
                # Per-action accuracy
                for i in range(5):  # 5 actions
                    action_correct[i] += (predicted[:, i] == actions[:, i]).sum().item()
                    action_total[i] += actions.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        exact_match_accuracy = exact_matches / total_samples
        
        # Calculate per-action accuracies
        per_action_accuracies = []
        for i in range(5):
            acc = action_correct[i].item() / action_total[i].item()
            per_action_accuracies.append(acc)
        
        return avg_val_loss, exact_match_accuracy, per_action_accuracies
    
    def plot_training_history(self):
        """Plot training metrics"""
        if not self.train_losses:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        axes[0, 0].plot(self.train_losses, label='Training Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Exact match accuracy
        axes[0, 1].plot(self.val_accuracies, label='Exact Match Accuracy', color='green')
        axes[0, 1].set_title('Exact Match Accuracy Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Per-action accuracy over time
        if self.per_action_accuracies:
            action_names = ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']
            per_action_array = np.array(self.per_action_accuracies)
            
            for i, action in enumerate(action_names):
                axes[1, 0].plot(per_action_array[:, i], label=action)
            
            axes[1, 0].set_title('Per-Action Accuracy Over Time')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Final per-action accuracy bar chart
        if self.per_action_accuracies:
            final_accuracies = self.per_action_accuracies[-1]
            action_names = ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']
            
            bars = axes[1, 1].bar(action_names, final_accuracies)
            axes[1, 1].set_title('Final Per-Action Accuracy')
            axes[1, 1].set_xlabel('Action')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, final_accuracies):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{acc:.3f}', ha='center', va='bottom')
            
            # Rotate x-labels for better readability
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Training history plot saved as 'training_history.png'")

    def train(self, train_loader, val_loader, epochs=50, learning_rate=0.001):
        """Train the behavioral cloning model for multi-label classification"""
        
        # Calculate class weights for each action independently
        print("\n=== Calculating per-action class weights ===")
        pos_weights = []
        
        for i, col in enumerate(['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']):
            positive_samples = train_loader.dataset.dataset.data[col].sum()
            total_samples = len(train_loader.dataset.dataset.data)
            negative_samples = total_samples - positive_samples
            
            if positive_samples > 0:
                # For multi-label: weight = negative_samples / positive_samples
                weight = negative_samples / positive_samples
                pos_weights.append(weight)
                print(f"{col:15s}: pos={positive_samples:5d}, neg={negative_samples:5d}, weight={weight:.2f}")
            else:
                pos_weights.append(1.0)
                print(f"{col:15s}: pos={positive_samples:5d}, neg={negative_samples:5d}, weight=1.00 (no positives)")
        
        pos_weights = torch.FloatTensor(pos_weights).to(self.device)
        
        # Use BCEWithLogitsLoss with pos_weight for multi-label classification
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        print(f"\nTraining on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_idx, (features, actions) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                features, actions = features.to(self.device), actions.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                
                # BCEWithLogitsLoss expects raw logits, not sigmoid outputs
                loss = criterion(outputs, actions)
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_loss, val_accuracy, per_action_acc = self.validate(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
            
            # Store history
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            self.per_action_accuracies.append(per_action_acc)
            
            # Enhanced logging
            per_action_str = ", ".join([f"{acc:.2f}" for acc in per_action_acc])
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Exact Match: {val_accuracy:.3f}")
            print(f"    Per-action accuracy: [{per_action_str}]")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # Load best model
        self.load_checkpoint('best_model.pth')
        print("Training completed!")

def export_model_for_inference(model, image_size, output_dir='model'):
    """
    Export model and preprocessing components for inference
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(output_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save model architecture info with updated action columns
    model_info = {
        'input_dim': model.network[0].in_features,
        'hidden_dims': [layer.out_features for layer in model.network if isinstance(layer, nn.Linear)][:-1],
        'output_dim': model.network[-2].out_features,
        'image_size': image_size,
        'action_columns': ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']
    }
    
    info_path = os.path.join(output_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model exported to {output_dir}/")
    print("Files created:")
    print("- model.pth: PyTorch model weights")
    print("- model_info.json: Model architecture and action mapping")

def load_model_for_inference(model_dir='model'):
    """
    Load model for inference (example usage)
    """
    # Load model info
    with open(os.path.join(model_dir, 'model_info.json'), 'r') as f:
        model_info = json.load(f)
    
    # Recreate model
    model = BehavioralCloningNet(
        input_dim=model_info['input_dim'],
        hidden_dims=model_info['hidden_dims'],
        output_dim=model_info['output_dim']
    )
    
    # Load weights
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), 
                                   map_location='cpu'))
    
    model.eval()
    return model, model_info['image_size'], model_info['action_columns']

def main():
    print("Script started!")
    """Main training script"""
    # Configuration
    DATA_DIR = r"C:\Users\Abdullah\Downloads\HKData"
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    IMAGE_SIZE = (80, 60)  # Smaller size for faster training
    
    # Find latest session data
    csv_files = [f for f in os.listdir(DATA_DIR) if f.startswith('hk_actions_') and f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found!")
        return
    
    latest_csv = max(csv_files)
    session_id = latest_csv.replace('hk_actions_', '').replace('.csv', '')
    frames_dir = os.path.join(DATA_DIR, f'frames_{session_id}')
    
    print(f"Training on session: {session_id}")
    print(f"CSV: {latest_csv}")
    print(f"Frames: {frames_dir}")
    
    if not os.path.exists(frames_dir):
        print(f"Frames directory not found: {frames_dir}")
        return
    
    # Create dataset
    dataset = HollowKnightDataset(
        csv_path=os.path.join(DATA_DIR, latest_csv),
        frames_path=frames_dir,
        image_size=IMAGE_SIZE
    )
    
    if len(dataset) == 0:
        print("No valid data found!")
        return
    
    # Add detailed analysis of action distribution
    print("\n=== Detailed Action Analysis ===")
    df = dataset.data
    for col in dataset.action_columns:
        true_count = df[col].sum()
        false_count = len(df) - true_count
        ratio = true_count / false_count if false_count > 0 else float('inf')
        print(f"{col:15s}: True={true_count:5d}, False={false_count:5d}, Ratio={ratio:.3f}")
    
    # Check for data quality issues
    print("\n=== Data Quality Check ===")
    # Check if player is always moving left
    always_left = df['moving_left'].all()
    never_right = not df['moving_right'].any()
    print(f"Always moving left: {always_left}")
    print(f"Never moving right: {never_right}")
    
    # Check position variance
    pos_variance = df[['x_position', 'y_position']].var()
    print(f"Position variance - X: {pos_variance['x_position']:.2f}, Y: {pos_variance['y_position']:.2f}")
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model - input dimension now includes image + 4 position features
    input_dim = IMAGE_SIZE[0] * IMAGE_SIZE[1] + 4  # Flattened image + player_x, player_y, enemy_x, enemy_y
    model = BehavioralCloningNet(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        output_dim=len(dataset.action_columns),
        dropout_rate=0.5  # Increase dropout to reduce overfitting
    )
    
    print(f"Model input dimension: {input_dim}")
    
    # Create trainer
    trainer = BehavioralCloningTrainer(model)
    
    # Train model with adjusted learning rate
    trainer.train(train_loader, val_loader, epochs=EPOCHS, learning_rate=0.0005)  # Lower learning rate
    
    # Plot training history
    trainer.plot_training_history()
    
    # Export model for inference
    export_model_for_inference(model, IMAGE_SIZE)
    
    print("Training completed and model exported!")
    
    # Test inference (example)
    print("\nTesting inference...")
    loaded_model, image_size, action_columns = load_model_for_inference()
    print(f"Model loaded successfully!")
    print(f"Image size: {image_size}")
    print(f"Action columns: {action_columns}")

    if os.path.exists('best_model.pth'):
        os.remove('best_model.pth')
    print("Cleaned up temporary checkpoint file")


if __name__ == "__main__":
    main()