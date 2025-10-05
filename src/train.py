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
        Neural network for behavioral cloning
        
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
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Sigmoid for binary actions
        
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
    
    def train(self, train_loader, val_loader, epochs=50, learning_rate=0.001):
        """Train the behavioral cloning model"""
        
        # Calculate class weights to handle imbalance
        print("\n=== Calculating class weights ===")
        action_sums = []
        for col in ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']:
            action_sum = train_loader.dataset.dataset.data[col].sum()
            action_sums.append(action_sum)
        
        total_samples = len(train_loader.dataset.dataset.data)
        pos_weights = []
        for i, action_sum in enumerate(action_sums):
            if action_sum > 0:
                # Weight = total_negative / total_positive
                neg_samples = total_samples - action_sum
                weight = neg_samples / action_sum
                pos_weights.append(weight)
                print(f"Action {i}: weight = {weight:.2f}")
            else:
                pos_weights.append(1.0)
        
        pos_weights = torch.FloatTensor(pos_weights).to(self.device)
        
        # Use weighted BCE loss
        criterion = nn.BCELoss(reduction='none')
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
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
                
                # Apply weighted loss
                loss_per_sample = criterion(outputs, actions)
                weighted_loss = loss_per_sample * pos_weights.unsqueeze(0)
                loss = weighted_loss.mean()
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_loss, val_accuracy = self.validate(val_loader, criterion, pos_weights)
            
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
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Exact Match Acc: {val_accuracy:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # Load best model
        self.load_checkpoint('best_model.pth')
        print("Training completed!")
    
    def validate(self, val_loader, criterion, pos_weights):
        """Validate the model - exact match accuracy"""
        self.model.eval()
        val_loss = 0
        exact_matches = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, actions in val_loader:
                features, actions = features.to(self.device), actions.to(self.device)
                outputs = self.model(features)
                
                # Calculate weighted loss
                loss_per_sample = criterion(outputs, actions)
                weighted_loss = loss_per_sample * pos_weights.unsqueeze(0)
                loss = weighted_loss.mean()
                val_loss += loss.item()
                
                # Calculate exact match accuracy (all actions must match)
                predicted = (outputs > 0.5).float()
                exact_match = (predicted == actions).all(dim=1).float()
                exact_matches += exact_match.sum().item()
                total_samples += actions.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = exact_matches / total_samples
        
        return avg_val_loss, accuracy
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }, filepath)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.val_accuracies, label='Exact Match Accuracy')
        ax2.set_title('Validation Exact Match Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

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
        output_dim=len(dataset.action_columns)  # Now 5 actions instead of 9
    )
    
    print(f"Model input dimension: {input_dim}")
    
    # Create trainer
    trainer = BehavioralCloningTrainer(model)
    
    # Train model
    trainer.train(train_loader, val_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    
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