# Standard library imports
import os
import json
import random
import warnings
import datetime
import copy

# Third-party imports
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Scikit-learn imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_recall_curve

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp

# Warnings configuration
warnings.filterwarnings('ignore')

def engineer_features(df):
    """
    Adds derived features to the dataframe.
    """
    df['player_dx'] = df['x_position'].diff().fillna(0)
    df['player_dy'] = df['y_position'].diff().fillna(0)
    df['enemy_dx'] = df['enemy_x'].diff().fillna(0)
    df['enemy_dy'] = df['enemy_y'].diff().fillna(0)
    df['player_speed'] = np.sqrt(df['player_dx']**2 + df['player_dy']**2)
    df['enemy_distance'] = np.sqrt((df['x_position'] - df['enemy_x'])**2 + (df['y_position'] - df['enemy_y'])**2)
    
    action_cols = ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']
    for col in action_cols:
        df[f'prev_{col}'] = df[col].shift(1).fillna(0)
        
    return df

class HollowKnightDataset(Dataset):
    def __init__(self, data_df, transform=None, image_size=(80, 60), n_frames=1, feature_columns=None, action_columns=None):
        self.data = data_df.copy().reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size
        self.n_frames = n_frames
        self.action_columns = action_columns or ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']
        self.feature_columns = feature_columns or ['x_position', 'y_position', 'enemy_x', 'enemy_y']
        
        missing_cols = [c for c in self.feature_columns if c not in self.data.columns]
        if missing_cols:
             # Just a warning or fill with 0 to allow flexibility if user didn't engineer everything
             # But strictly better to raise error if critical
             pass

        if not self.image_size:
            self.image_size = self._get_original_image_size()
        
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
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        current_frame_id = row['frame_id']
        current_frame_path = row['frame_path']
        frames = []
        for i in range(self.n_frames):
            path = self._get_previous_frame_path(current_frame_path, current_frame_id, i) if i > 0 else current_frame_path
            if os.path.exists(path):
                with Image.open(path).convert('L') as img:
                    if self.image_size:
                        img = img.resize(self.image_size, Image.Resampling.LANCZOS)
                    frames.append(transforms.ToTensor()(img))
            else:
                frames.append(frames[-1] if frames else torch.zeros(1, self.image_size[1], self.image_size[0]))

        image_stack = torch.cat(frames, dim=0)
        if self.transform: image_stack = self.transform(image_stack)
        other_features = torch.FloatTensor([float(row[col]) for col in self.feature_columns])
        actions = torch.FloatTensor([float(row[col]) for col in self.action_columns])
        return image_stack, other_features, actions

class BehavioralCloningNet(nn.Module):
    def __init__(self, image_shape=(1, 60, 80), other_features_dim=4, num_actions=5, dropout_rate=0.5):
        super(BehavioralCloningNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=image_shape[0], out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._conv_out_shape = self._get_conv_out_shape(image_shape)
        self.fc1 = nn.Linear(self._conv_out_shape + other_features_dim, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_actions)

    def _get_conv_out_shape(self, shape):
        with torch.no_grad():
            dummy_tensor = torch.zeros(1, *shape)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy_tensor))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            return int(np.prod(x.shape))

    def forward(self, image, other_features):
        x = self.pool(F.relu(self.bn1(self.conv1(image))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        combined = torch.cat([x, other_features], dim=1)
        combined = F.relu(self.bn3(self.fc1(combined)))
        combined = self.dropout(combined)
        return self.fc2(combined)

class MetaLearner(nn.Module):
    def __init__(self, input_dim=10, output_dim=5):
        super(MetaLearner, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc(x)
    def get_l1_loss(self): return torch.norm(self.fc.weight, 1)
    def get_l2_loss(self): return torch.norm(self.fc.weight, 2)

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
        self.scaler = amp.GradScaler()
        self.reset_history()

    def reset_history(self):
        self.train_losses, self.val_losses, self.val_accuracies, self.partial_accuracies, self.per_action_accuracies = [], [], [], [], []
    
    def save_checkpoint(self, filepath): torch.save(self.model.state_dict(), filepath)
    def load_checkpoint(self, filepath): self.model.load_state_dict(torch.load(filepath, map_location=self.device))
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        val_loss, exact_matches, total_samples = 0, 0, 0
        total_correct_bits = 0
        num_actions = self.model.fc2.out_features
        action_correct, action_total = torch.zeros(num_actions, device=self.device), torch.zeros(num_actions, device=self.device)
        all_labels, all_preds = [], []

        with torch.no_grad():
            for images, other_features, actions in val_loader:
                images, other_features, actions = images.to(self.device), other_features.to(self.device), actions.to(self.device)
                with amp.autocast(enabled=True):
                    outputs = self.model(images, other_features)
                    loss = criterion(outputs, actions)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                all_labels.append(actions.cpu()); all_preds.append(probs.cpu())
                predicted = (probs > 0.5).float()
                exact_matches += (predicted == actions).all(dim=1).sum().item()
                total_correct_bits += (predicted == actions).sum().item()
                total_samples += actions.size(0)
                for i in range(num_actions):
                    action_correct[i] += (predicted[:, i] == actions[:, i]).sum().item()
                    action_total[i] += actions.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        exact_match_accuracy = exact_matches / total_samples
        partial_match_accuracy = total_correct_bits / (total_samples * num_actions)
        per_action_acc = [corr.item() / total.item() if total.item() > 0 else 0 for corr, total in zip(action_correct, action_total)]
        return avg_val_loss, exact_match_accuracy, partial_match_accuracy, per_action_acc, torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()

    def plot_training_history(self, action_names, save_path='training_history.png'):
        if not self.train_losses: return
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(self.train_losses, label='Training Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].grid(True); axes[0, 0].legend()
        
        axes[0, 1].plot(self.val_accuracies, label='Exact Match Accuracy', color='green')
        axes[0, 1].plot(self.partial_accuracies, label='Partial Match Accuracy', color='orange', linestyle='--')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].grid(True); axes[0, 1].legend()

        per_action_array = np.array(self.per_action_accuracies)
        for i, action in enumerate(action_names):
            axes[1, 0].plot(per_action_array[:, i], label=action)
        axes[1, 0].set_title('Per-Action Accuracy')
        axes[1, 0].grid(True); axes[1, 0].legend()

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
        action_columns = train_loader.dataset.action_columns
        pos_weights = []
        for col in action_columns:
            pos = train_loader.dataset.data[col].sum()
            pos_weights.append((len(train_loader.dataset.data) - pos) / (pos + 1))
        
        criterion = FocalLoss(pos_weight=torch.FloatTensor(pos_weights).to(self.device)) if loss_type == 'focal' else nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(pos_weights).to(self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for images, other_features, actions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, other_features, actions = images.to(self.device), other_features.to(self.device), actions.to(self.device)
                optimizer.zero_grad()
                with amp.autocast(enabled=True):
                    outputs = self.model(images, other_features)
                    loss = criterion(outputs, actions)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer); self.scaler.update()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            val_loss, val_acc, partial_acc, per_action_acc, _, _ = self.validate(val_loader, criterion)
            
            # Record history
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.partial_accuracies.append(partial_acc)
            self.per_action_accuracies.append(per_action_acc)

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss; patience_counter = 0; self.save_checkpoint('best_model.pth')
            else: patience_counter += 1
            
            per_action_str = ", ".join([f"{a:.3f}" for a in per_action_acc])
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.3f} | Partial: {partial_acc:.3f} | Per-Action: [{per_action_str}]")
            if patience_counter >= 15: break
        self.load_checkpoint('best_model.pth')

def export_model_for_inference(model, image_size, n_frames, feature_columns, output_dir='model'):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    model_info = {
        'image_shape': [n_frames, image_size[1], image_size[0]],
        'other_features_dim': len(feature_columns),
        'feature_columns': feature_columns,
        'action_columns': ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing'][:model.fc2.out_features],
        'n_frames': n_frames
    }
    with open(os.path.join(output_dir, 'model_info.json'), 'w') as f: json.dump(model_info, f, indent=2)

def load_model_for_inference(model_dir='model'):
    with open(os.path.join(model_dir, 'model_info.json'), 'r') as f: info = json.load(f)
    model = BehavioralCloningNet(image_shape=tuple(info['image_shape']), other_features_dim=info['other_features_dim'], num_actions=len(info.get('action_columns', [1]*5)))
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location='cpu'))
    return model.eval(), info

def main():
    DATA_DIR = r"C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\HKData"
    STACK_SIZE = 3
    BASE_LOG_DIR = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(BASE_LOG_DIR, exist_ok=True)

    csv_files = [f for f in os.listdir(DATA_DIR) if f.startswith('hk_actions_') and f.endswith('.csv')]
    all_dfs = []
    for csv_file in tqdm(csv_files, desc="Loading Data"):
        sid = csv_file.replace('hk_actions_', '').replace('.csv', '')
        frames_dir = os.path.join(DATA_DIR, f'frames_{sid}')
        if os.path.exists(frames_dir):
            df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
            df['frame_path'] = df['frame_id'].apply(lambda x: os.path.join(frames_dir, f"frame_{x:06d}.png"))
            df = engineer_features(df[df['frame_path'].apply(os.path.exists)].copy())
            all_dfs.append(df)
    
    master_df = pd.concat(all_dfs, ignore_index=True)
    all_action_cols = ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']
    
    # Define specialized features for each expert
    global_features = [
        'x_position', 'y_position', 'enemy_x', 'enemy_y', 
        'player_dx', 'player_dy', 'enemy_dx', 'enemy_dy', 
        'player_speed', 'enemy_distance',                 
        'prev_moving_left', 'prev_moving_right', 'prev_attacking', 'prev_jumping', 'prev_dashing' 
    ]
    
    expert_configs = [
        {'name': 'global', 'actions': all_action_cols, 'features': global_features},
        # Movement expert now includes enemy features as requested
        {'name': 'movement', 'actions': ['moving_left', 'moving_right'], 'features': ['x_position', 'y_position', 'player_speed', 'enemy_x', 'enemy_y', 'enemy_distance', 'prev_moving_left', 'prev_moving_right']},
        {'name': 'jump_dash', 'actions': ['jumping', 'dashing'], 'features': ['y_position', 'player_dy', 'enemy_distance', 'prev_jumping', 'prev_dashing']},
        {'name': 'combat', 'actions': ['attacking'], 'features': ['enemy_x', 'enemy_y', 'enemy_distance', 'prev_attacking']}
    ]

    # Meta-context features that help the meta-learner decide who to trust
    meta_context_features = ['enemy_distance', 'player_speed', 'player_y']

    train_df, val_df = train_test_split(master_df, test_size=0.2, random_state=42)
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    expert_outputs_train, expert_outputs_val = [], []

    for config in expert_configs:
        print(f"\n--- Training Expert: {config['name']} ---")
        train_ds = HollowKnightDataset(train_df, image_size=(320, 160), n_frames=STACK_SIZE, feature_columns=config['features'], action_columns=config['actions'])
        val_ds = HollowKnightDataset(val_df, image_size=(320, 160), n_frames=STACK_SIZE, feature_columns=config['features'], action_columns=config['actions'])
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        
        model = BehavioralCloningNet(image_shape=(STACK_SIZE, 160, 320), other_features_dim=len(config['features']), num_actions=len(config['actions']))
        trainer = BehavioralCloningTrainer(model, log_dir=os.path.join(BASE_LOG_DIR, f"expert_{config['name']}"))
        trainer.train(train_loader, val_loader, epochs=100, learning_rate=1e-4, weight_decay=1e-4, loss_type='focal')
        
        output_dir = f"model/ensemble_{config['name']}"
        export_model_for_inference(model, (320, 160), STACK_SIZE, config['features'], output_dir=output_dir)
        trainer.plot_training_history(config['actions'], save_path=os.path.join(output_dir, 'training_history.png'))
        
        def collect(loader):
            model.eval()
            outs = []
            with torch.no_grad():
                for img, feat, _ in tqdm(loader): outs.append(model(img.to(trainer.device), feat.to(trainer.device)).cpu())
            return torch.cat(outs)

        # For Meta-Learner training, we need predictions on the training set (no shuffle)
        expert_outputs_train.append(collect(DataLoader(train_ds, batch_size=64, shuffle=False)))
        expert_outputs_val.append(collect(val_loader))

    print("\n--- Tuning Meta-Learner (Random Search CV) ---")
    context_train = torch.FloatTensor(train_df[meta_context_features].values)
    context_val = torch.FloatTensor(val_df[meta_context_features].values)
    
    # Input = Expert Logits + Meta Context
    X_meta_all = torch.cat(expert_outputs_train + [context_train], dim=1)
    y_meta_all = torch.FloatTensor(train_df[all_action_cols].values)

    best_hp, best_loss = None, float('inf')
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    # Random Search for Hyperparameters (hp)
    for trial in range(15):
        hp = {
            'lr': random.choice([1e-2, 1e-3, 5e-4]), 
            'l1': random.choice([0, 1e-4, 1e-3]), 
            'l2': random.choice([1e-4, 1e-3, 1e-2]), 
            'epochs': 100
        }
        fold_losses = []
        for tr_idx, vl_idx in kf.split(X_meta_all):
            m = MetaLearner(input_dim=X_meta_all.shape[1], output_dim=5)
            opt = optim.Adam(m.parameters(), lr=hp['lr'])
            for _ in range(hp['epochs']):
                m.train(); opt.zero_grad(); out = m(X_meta_all[tr_idx])
                loss = nn.BCEWithLogitsLoss()(out, y_meta_all[tr_idx]) + hp['l1']*m.get_l1_loss() + hp['l2']*m.get_l2_loss()
                loss.backward(); opt.step()
            m.eval(); fold_losses.append(nn.BCEWithLogitsLoss()(m(X_meta_all[vl_idx]), y_meta_all[vl_idx]).item())
        
        avg_loss = np.mean(fold_losses)
        if avg_loss < best_loss: best_loss = avg_loss; best_hp = hp
        print(f"Trial {trial+1}: CV Loss {avg_loss:.5f} | HP: {hp}")

    print(f"Best HP: {best_hp}")
    # --- Final Meta-Learner Training ---
    print("\n--- Training Final Meta-Learner ---")
    input_dim = X_meta_all.shape[1]
    final_meta_model = MetaLearner(input_dim=input_dim, output_dim=5)
    final_optimizer = optim.Adam(final_meta_model.parameters(), lr=best_hp['lr'])
    final_criterion = nn.BCEWithLogitsLoss()
    
    meta_train_losses = []
    
    final_meta_model.train()
    for epoch in range(best_hp['epochs']):
        final_optimizer.zero_grad()
        outputs = final_meta_model(X_meta_all)
        loss = final_criterion(outputs, y_meta_all)
        
        loss += best_hp['l1'] * final_meta_model.get_l1_loss()
        loss += best_hp['l2'] * final_meta_model.get_l2_loss()
        
        loss.backward()
        final_optimizer.step()
        
        meta_train_losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Final Train Epoch {epoch}: Loss {loss.item():.5f}")

    # Plot Meta-Learner History
    plt.figure(figsize=(10, 6))
    plt.plot(meta_train_losses, label='Meta-Learner Training Loss')
    plt.title('Meta-Learner Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE + L1/L2)')
    plt.legend()
    plt.grid(True)
    plt.savefig('model/meta_learner_history.png')
    plt.close()

    torch.save(final_meta_model.state_dict(), 'model/meta_learner.pth')
    with open('model/meta_info.json', 'w') as f: json.dump({'meta_features': meta_context_features}, f)
    print("\nTraining Complete!")

if __name__ == "__main__": main()