import torch
import torch.nn as nn
import numpy as np
import json
import cv2
import sys
import base64
from pathlib import Path
import pyautogui
import time
import threading

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
        
        # Remove the sigmoid - model outputs raw logits
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class HollowKnightAI:
    def __init__(self, model_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model info
        model_info_path = Path(model_dir) / 'model_info.json'
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # Load model
        self.model = BehavioralCloningNet(
            input_dim=self.model_info['input_dim'],
            hidden_dims=self.model_info['hidden_dims'],
            output_dim=self.model_info['output_dim'],
            dropout_rate=0.0  # No dropout during inference
        )
        
        # Load model weights - updated to match training export
        weights_path = Path(model_dir) / 'model.pth'
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.action_columns = self.model_info['action_columns']
        self.image_size = tuple(self.model_info['image_size'])  # (width, height)
        
        # Updated key mappings to match the 5 actions in the CSV
        self.action_keys = {
            'moving_left': 'left',
            'moving_right': 'right',
            'attacking': 'x',
            'jumping': 'z',
            'dashing': 'c'
        }
        
        # Track currently pressed keys
        self.pressed_keys = set()
        
        # Jump timing control
        self.jump_start_time = None
        self.jump_duration = 2.0  # 2 seconds
        
        print(f"Model loaded - Image size: {self.image_size}, Input dim: {self.model_info['input_dim']}")
        print(f"Action columns: {self.action_columns}")
    
    def _handle_jump_timing(self, should_jump):
        """Handle jump key timing - keep pressed for 2 seconds once triggered"""
        current_time = time.time()
        jump_key = self.action_keys['jumping']
        
        if should_jump and self.jump_start_time is None:
            # Start jump
            self.jump_start_time = current_time
            if jump_key not in self.pressed_keys:
                pyautogui.keyDown(jump_key)
                self.pressed_keys.add(jump_key)
        
        # Check if jump duration has elapsed
        if self.jump_start_time is not None:
            if current_time - self.jump_start_time >= self.jump_duration:
                # End jump
                if jump_key in self.pressed_keys:
                    pyautogui.keyUp(jump_key)
                    self.pressed_keys.discard(jump_key)
                self.jump_start_time = None
                return False  # Jump is no longer active
            else:
                return True  # Jump is still active
        
        return False
    




    def predict_from_bytes(self, image_bytes, width, height, player_x=0.5, player_y=0.5, enemy_x=0.0, enemy_y=0.0):
        """Run prediction on image bytes and press keys"""
        try:
            # Convert bytes to numpy array
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            image_np = image_np.reshape((height, width))
            
            # Resize to match training image size (width, height)
            image_resized = cv2.resize(image_np, self.image_size)
            
            # Normalize to [0, 1]
            image = image_resized.astype(np.float32) / 255.0
            
            # Flatten for neural network
            image_flat = image.flatten()
            
            # Add position features (normalized to [0, 1])
            position_features = np.array([player_x, player_y, enemy_x, enemy_y], dtype=np.float32)
            
            # Combine image and position features
            features = np.concatenate([image_flat, position_features])
            
            # Convert to tensor and add batch dimension
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(features_tensor)
                # Apply sigmoid to convert logits to probabilities
                predictions = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Enhanced debugging
            predictions_str = ", ".join([f"{action}:{pred:.3f}" for action, pred in zip(self.action_columns, predictions)])
            sys.stderr.write(f"Predictions: {predictions_str}\n")
            
            # Handle jump timing first
            should_jump = predictions[self.action_columns.index('jumping')] > 0.3 if 'jumping' in self.action_columns else False  # Lower jump threshold
            jump_is_active = self._handle_jump_timing(should_jump)
            
            # Use lower, action-specific thresholds based on your data distribution
            thresholds = {
                'moving_left': 0.25,    # 20.6% in data, so lower threshold
                'moving_right': 0.25,   # 22.1% in data, so lower threshold
                'attacking': 0.35,      # 28.1% in data, higher threshold
                'jumping': 0.3,         # 3.1% in data, much lower threshold
                'dashing': 0.1          # 0% in data, very low threshold to encourage
            }
            
            # Get keys to press
            keys_to_press = set()
            
            for i, (action, pred) in enumerate(zip(self.action_columns, predictions)):
                threshold = thresholds.get(action, 0.5)
                
                if action == 'jumping':
                    if jump_is_active:
                        keys_to_press.add(self.action_keys[action])
                        sys.stderr.write(f"Jump active (timing)\n")
                elif pred > threshold and action in self.action_keys:
                    keys_to_press.add(self.action_keys[action])
                    sys.stderr.write(f"Action triggered: {action} ({pred:.3f} > {threshold:.3f})\n")
            
            # Handle movement conflicts - your data shows these rarely happen together
            if 'left' in keys_to_press and 'right' in keys_to_press:
                left_pred = predictions[self.action_columns.index('moving_left')]
                right_pred = predictions[self.action_columns.index('moving_right')]
                
                # Keep the stronger prediction
                if left_pred > right_pred:
                    keys_to_press.discard('right')
                    sys.stderr.write(f"Conflict resolved: keeping left ({left_pred:.3f}) over right ({right_pred:.3f})\n")
                else:
                    keys_to_press.discard('left')
                    sys.stderr.write(f"Conflict resolved: keeping right ({right_pred:.3f}) over left ({left_pred:.3f})\n")
            
            # Anti-spam: only change if predictions are confident enough or significantly different
            if hasattr(self, 'last_predictions'):
                # Check if any action crossed its threshold significantly
                significant_change = False
                for i, (action, pred) in enumerate(zip(self.action_columns, predictions)):
                    last_pred = self.last_predictions[i]
                    threshold = thresholds.get(action, 0.5)
                    
                    # Check for threshold crossing
                    if (pred > threshold and last_pred <= threshold) or (pred <= threshold and last_pred > threshold):
                        significant_change = True
                        sys.stderr.write(f"Significant change in {action}: {last_pred:.3f} -> {pred:.3f} (threshold: {threshold:.3f})\n")
                        break
                
                if not significant_change:
                    # Keep current state if no significant changes
                    sys.stderr.write("No significant prediction changes, maintaining current state\n")
                    return list(self.pressed_keys)
            
            self.last_predictions = predictions.copy()
            
            # Release keys that should no longer be pressed (excluding jump)
            keys_to_release = self.pressed_keys - keys_to_press
            jump_key = self.action_keys['jumping']
            keys_to_release.discard(jump_key)  # Don't auto-release jump key
            
            for key in keys_to_release:
                pyautogui.keyUp(key)
                sys.stderr.write(f"Released key: {key}\n")
            
            # Press new keys (excluding jump which is handled separately)
            keys_to_press_new = keys_to_press - self.pressed_keys
            if jump_key in keys_to_press_new:
                keys_to_press_new.discard(jump_key)  # Jump is handled separately
            
            for key in keys_to_press_new:
                pyautogui.keyDown(key)
                sys.stderr.write(f"Pressed key: {key}\n")
            
            # Update pressed keys
            self.pressed_keys = keys_to_press
            
            # Log final action
            if keys_to_press:
                actions_str = ", ".join([action for action, key in self.action_keys.items() if key in keys_to_press])
                sys.stderr.write(f"Final actions: {actions_str}\n")
            else:
                sys.stderr.write("Final actions: No actions (idle)\n")
            
            return list(keys_to_press)
            
        except Exception as e:
            print(f"ERROR:{e}", file=sys.stderr)
            sys.stderr.flush()
            return []


    def release_all_keys(self):
        """Release all currently pressed keys"""
        for key in self.pressed_keys:
            try:
                pyautogui.keyUp(key)
            except:
                pass
        self.pressed_keys.clear()
        self.jump_start_time = None


def main():
    """Main inference loop"""
    # Default model directory - update as needed
    model_dir = r"C:\Users\Abdullah\Documents\GitHub\Hollow-Knight-AI\model"  # Or get from command line arguments
    
    try:
        # Initialize AI
        ai = HollowKnightAI(model_dir)
        print("READY")
        sys.stdout.flush()
        
        # Command loop
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                    
                line = line.strip()
                
                if line == "QUIT":
                    ai.release_all_keys()  # Clean up before quitting
                    print("QUIT_OK")
                    sys.stdout.flush()
                    break
                
                elif line.startswith("PREDICT:"):
                    # Parse: PREDICT:width:height:base64data or PREDICT:width:height:player_x:player_y:enemy_x:enemy_y:base64data
                    parts = line.split(':', 7)
                    
                    if len(parts) == 4:
                        # Old format: PREDICT:width:height:base64data
                        width = int(parts[1])
                        height = int(parts[2])
                        base64_data = parts[3]
                        player_x = player_y = 0.5  # Default center position
                        enemy_x = enemy_y = 0.0    # Default no enemy
                        
                    elif len(parts) == 8:
                        # New format: PREDICT:width:height:player_x:player_y:enemy_x:enemy_y:base64data
                        width = int(parts[1])
                        height = int(parts[2])
                        player_x = float(parts[3])
                        player_y = float(parts[4])
                        enemy_x = float(parts[5])
                        enemy_y = float(parts[6])
                        base64_data = parts[7]
                        
                    else:
                        print("ERROR:Invalid format")
                        sys.stdout.flush()
                        continue
                    
                    # Decode image
                    image_bytes = base64.b64decode(base64_data)
                    
                    # Get prediction
                    keys = ai.predict_from_bytes(image_bytes, width, height, player_x, player_y, enemy_x, enemy_y)
                    
                    # Send back keys as comma-separated string
                    if keys:
                        print(f"KEYS:{','.join(keys)}")
                    else:
                        print("KEYS:")
                    sys.stdout.flush()
                    
            except Exception as e:
                print(f"ERROR:{e}")
                sys.stdout.flush()
                
    except Exception as e:
        print(f"FATAL:{e}")
        sys.stderr.flush()

if __name__ == "__main__":
    main()