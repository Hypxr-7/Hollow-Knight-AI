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

class BehavioralCloningNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], output_dim=9, dropout_rate=0.3):
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
        layers.append(nn.Sigmoid())
        
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
        
        # Key mappings
        self.action_keys = {
            'moving_left': 'left',
            'moving_right': 'right', 
            'moving_up': 'up',
            'moving_down': 'down',
            'attacking': 'z',
            'jumping': 'x',
            'dashing': 'c',
            'focusing': 'a',
            'dreamnail': 'q'
        }
        
        # Track currently pressed keys
        self.pressed_keys = set()
        
        print(f"Model loaded - Image size: {self.image_size}, Input dim: {self.model_info['input_dim']}")
    
    def predict_from_bytes(self, image_bytes, width, height):
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
            
            # Convert to tensor and add batch dimension
            image_tensor = torch.FloatTensor(image_flat).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                predictions = self.model(image_tensor).cpu().numpy()[0]
            
            # Log predictions for debugging
            predictions_str = ", ".join([f"{action}:{pred:.3f}" for action, pred in zip(self.action_columns, predictions)])
            sys.stderr.write(f"Predictions: {predictions_str}\n")
            sys.stderr.flush()
            
            # Get keys to press (threshold 0.5)
            keys_to_press = set()
            for action, pred in zip(self.action_columns, predictions):
                if pred > 0.5 and action in self.action_keys:
                    keys_to_press.add(self.action_keys[action])
            
            # Release keys that should no longer be pressed
            keys_to_release = self.pressed_keys - keys_to_press
            for key in keys_to_release:
                pyautogui.keyUp(key)
            
            # Press new keys
            keys_to_press_new = keys_to_press - self.pressed_keys
            for key in keys_to_press_new:
                pyautogui.keyDown(key)
            
            # Update pressed keys
            self.pressed_keys = keys_to_press
            
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
                    # Parse: PREDICT:width:height:base64data
                    parts = line.split(':', 3)
                    if len(parts) != 4:
                        print("ERROR:Invalid format")
                        sys.stdout.flush()
                        continue
                    
                    width = int(parts[1])
                    height = int(parts[2])
                    base64_data = parts[3]
                    
                    # Decode image
                    image_bytes = base64.b64decode(base64_data)
                    
                    # Get prediction
                    keys = ai.predict_from_bytes(image_bytes, width, height)
                    
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