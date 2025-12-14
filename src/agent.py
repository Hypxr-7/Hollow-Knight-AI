import torch
import numpy as np
import json
import cv2
import sys
import base64
import os
import pyautogui
import time

# Import the correct, up-to-date classes and functions from train.py
from train import BehavioralCloningNet, load_model_for_inference

class EnsembleAgent:
    """
    An agent that uses an ensemble of CNN models to make predictions.
    """
    def __init__(self, model_base_dir='model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_with_info = []

        print("Loading ensemble models...", file=sys.stderr)
        
        ensemble_dirs = sorted([
            os.path.join(model_base_dir, d) for d in os.listdir(model_base_dir) 
            if d.startswith('ensemble_') and os.path.isdir(os.path.join(model_base_dir, d))
        ])
        
        if not ensemble_dirs:
            raise FileNotFoundError(f"No 'ensemble_*' directories found in '{model_base_dir}'")

        print(f"Found {len(ensemble_dirs)} models in the ensemble.", file=sys.stderr)

        for model_dir in ensemble_dirs:
            print(f"Loading model from {model_dir}...", file=sys.stderr)
            model, info = load_model_for_inference(model_dir)
            model.to(self.device)
            self.models_with_info.append((model, info))
        
        self.action_columns = self.models_with_info[0][1]['action_columns']

        self.action_keys = {
            'moving_left': 'left', 'moving_right': 'right',
            'attacking': 'x', 'jumping': 'z', 'dashing': 'c'
        }
        self.pressed_keys = set()
        
        # Optimize PyAutoGUI speed
        pyautogui.PAUSE = 0
        
        print(f"Ensemble loaded with {len(self.models_with_info)} models.")

    def predict_from_bytes(self, image_bytes, width, height, player_x=0.5, player_y=0.5, enemy_x=0.0, enemy_y=0.0):
        try:
            # --- Preprocessing for CNN ---
            image_np = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width))

            other_features = torch.FloatTensor([
                player_x, player_y, enemy_x, enemy_y
            ]).unsqueeze(0).to(self.device)

            # --- Ensemble Prediction ---
            all_logits = []
            with torch.no_grad():
                for model, info in self.models_with_info:
                    # --- Preprocessing for each model ---
                    image_shape = info['image_shape']
                    image_size = (image_shape[2], image_shape[1]) # W, H

                    image_resized = cv2.resize(image_np, image_size, interpolation=cv2.INTER_AREA)
                    
                    image_tensor = torch.from_numpy(image_resized).float().to(self.device) / 255.0
                    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

                    logits = model(image_tensor, other_features)
                    all_logits.append(logits)
            
            avg_logits = torch.stack(all_logits).mean(dim=0)
            predictions = torch.sigmoid(avg_logits).cpu().numpy()[0]
            
            # --- Action Logic ---
            self.execute_actions(predictions)
            
        except Exception as e:
            print(f"ERROR in prediction: {e}", file=sys.stderr)
            self.release_all_keys()

    def execute_actions(self, predictions):
        """Determine which keys to press or release based on predictions."""
        # Actions that should be single presses, not holds (e.g., attacking, dashing)
        press_action_names = {'attacking', 'dashing'}

        desired_holds = set()
        
        thresholds = {
            'moving_left': 0.3, 'moving_right': 0.3,
            'attacking': 0.4, 'jumping': 0.35, 'dashing': 0.3
        }

        # Determine press and hold actions from predictions
        for i, action in enumerate(self.action_columns):
            if predictions[i] > thresholds.get(action, 0.5):
                action_key = self.action_keys[action]
                if action in press_action_names:
                    # For actions like attacking, press and release immediately.
                    # This prevents the key from being held down across multiple frames.
                    pyautogui.press(action_key) 
                else:
                    # For actions like moving or jumping, add to the set of keys to be held down.
                    desired_holds.add(action_key)

        # Handle mutual exclusion for movement
        if 'left' in desired_holds and 'right' in desired_holds:
            if predictions[self.action_columns.index('moving_left')] > predictions[self.action_columns.index('moving_right')]:
                desired_holds.discard('right')
            else:
                desired_holds.discard('left')
        
        # Update held keys based on the desired holds for this frame
        # First, release keys that are no longer desired to be held.
        keys_to_release = self.pressed_keys - desired_holds
        for key in keys_to_release:
            pyautogui.keyUp(key)
        
        # Then, press down new keys that should now be held.
        keys_to_press_new = desired_holds - self.pressed_keys
        for key in keys_to_press_new:
            pyautogui.keyDown(key)
            
        # Update the set of currently held keys for the next frame.
        # This set only contains keys for 'hold' actions.
        self.pressed_keys = desired_holds

    def release_all_keys(self):
        for key in list(self.pressed_keys):
            pyautogui.keyUp(key)
        self.pressed_keys.clear()

def main():
    """Main inference loop for the ensemble agent."""
    if len(sys.argv) > 1:
        model_base_dir = sys.argv[1]
    else:
        model_base_dir = 'model'
    try:
        agent = EnsembleAgent(model_base_dir)
        print("READY")
        sys.stdout.flush()
        
        while True:
            line = sys.stdin.readline()
            if not line: break
            line = line.strip()

            if line == "QUIT":
                agent.release_all_keys()
                print("QUIT_OK")
                sys.stdout.flush()
                break
            
            elif line.startswith("PREDICT:"):
                try:
                    # The C# mod sends: PREDICT:width:height:base64data
                    parts = line.split(':', 3)
                    if len(parts) == 4:
                        width, height = int(parts[1]), int(parts[2])
                        image_bytes = base64.b64decode(parts[3])
                        
                        # Call predict with default coords, as they are not sent from the mod
                        agent.predict_from_bytes(image_bytes, width, height)
                        print("KEYS:") # Acknowledge prediction
                        sys.stdout.flush()
                    else:
                        raise ValueError(f"Expected 4 parts but got {len(parts)}")

                except (ValueError, IndexError) as e:
                    print(f"ERROR:Invalid PREDICT format - {e}", file=sys.stderr)
                    sys.stdout.flush()
                    continue
                
    except Exception as e:
        print(f"FATAL:{e}", file=sys.stderr)
        sys.stderr.flush()

if __name__ == "__main__":
    main()