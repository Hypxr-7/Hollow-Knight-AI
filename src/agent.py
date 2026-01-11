import torch
import numpy as np
import json
import cv2
import sys
import base64
import os
import pyautogui
import time
from collections import deque

from train import BehavioralCloningNet, load_model_for_inference

class EnsembleAgent:
    """
    An agent that uses an ensemble of CNN models to make predictions.
    Supports frame stacking and stateful feature engineering.
    """
    def __init__(self, model_base_dir='model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Agent initialized on device: {self.device}", file=sys.stderr)
        self.models_with_info = []

        print("Loading ensemble models...", file=sys.stderr)
        
        # Find all ensemble directories
        ensemble_dirs = sorted([
            os.path.join(model_base_dir, d) for d in os.listdir(model_base_dir) 
            if d.startswith('ensemble_') and os.path.isdir(os.path.join(model_base_dir, d))
        ])
        
        if not ensemble_dirs:
            raise FileNotFoundError(f"No 'ensemble_*' directories found in '{model_base_dir}'")

        print(f"Found {len(ensemble_dirs)} models in the ensemble.", file=sys.stderr)

        # Load models and determine requirements
        self.max_frames_needed = 1
        
        for model_dir in ensemble_dirs:
            print(f"Loading model from {model_dir}...", file=sys.stderr)
            model, info, scaler = load_model_for_inference(model_dir)
            model.to(self.device)
            self.models_with_info.append((model, info, scaler))
            
            n_frames = info.get('n_frames', 1)
            if 'image_shape' in info and len(info['image_shape']) == 3:
                n_frames = max(n_frames, info['image_shape'][0])
            
            self.max_frames_needed = max(self.max_frames_needed, n_frames)
        
        # Buffer to store the last N raw frames
        self.frame_history = deque(maxlen=self.max_frames_needed)
        
        # State tracking for feature engineering
        self.last_player_x = None
        self.last_player_y = None
        self.last_enemy_x = None
        self.last_enemy_y = None
        self.prev_actions = {
            'moving_left': 0.0, 'moving_right': 0.0, 
            'moving_up': 0.0, 'moving_down': 0.0,
            'attacking': 0.0, 'jumping': 0.0, 'dashing': 0.0
        }
        
        # Maintain history of raw metrics for 3-frame scalar stacking (locality for 3)
        self.raw_history = deque(maxlen=3) # Stores dicts of base features
        
        self.action_columns = self.models_with_info[0][1]['action_columns']

        self.action_keys = {
            'moving_left': 'left', 'moving_right': 'right',
            'moving_up': 'up', 'moving_down': 'down',
            'attacking': 'x', 'jumping': 'z', 'dashing': 'c'
        }
        self.pressed_keys = set()
        
        # Optimize PyAutoGUI speed
        pyautogui.PAUSE = 0
        
        print(f"Ensemble loaded with {len(self.models_with_info)} models. Max history: {self.max_frames_needed} frames.")

    def _engineer_features_inference(self, player_x, player_y, enemy_x, enemy_y, scaler=None):
        """
        Replicates the logic from train.py's engineer_features function in a stateful way.
        """
        # Initialize previous state on first frame
        if self.last_player_x is None:
            self.last_player_x = player_x
            self.last_player_y = player_y
            self.last_enemy_x = enemy_x
            self.last_enemy_y = enemy_y

        # 1. Deltas
        player_dx = player_x - self.last_player_x
        player_dy = player_y - self.last_player_y
        enemy_dx = enemy_x - self.last_enemy_x
        enemy_dy = enemy_y - self.last_enemy_y
        
        # 2. Speed and Distance
        player_speed = np.sqrt(player_dx**2 + player_dy**2)
        enemy_distance = np.sqrt((player_x - enemy_x)**2 + (player_y - enemy_y)**2)
        
        # 3. Relative
        rel_x = enemy_x - player_x
        rel_y = enemy_y - player_y
        vel_diff_x = player_dx - enemy_dx
        vel_diff_y = player_dy - enemy_dy
        
        # 4. Current Raw Data Point
        current_raw = {
            'x_position': player_x, 'y_position': player_y, 'enemy_x': enemy_x, 'enemy_y': enemy_y,
            'player_dx': player_dx, 'player_dy': player_dy, 'enemy_dx': enemy_dx, 'enemy_dy': enemy_dy,
            'player_speed': player_speed, 'enemy_distance': enemy_distance,
            'rel_x': rel_x, 'rel_y': rel_y, 'vel_diff_x': vel_diff_x, 'vel_diff_y': vel_diff_y,
            'moving_left': self.prev_actions['moving_left'], 
            'moving_right': self.prev_actions['moving_right'],
            'moving_up': self.prev_actions['moving_up'],
            'moving_down': self.prev_actions['moving_down'],
            'attacking': self.prev_actions['attacking'], 
            'jumping': self.prev_actions['jumping'], 
            'dashing': self.prev_actions['dashing']
        }
        
        # Update raw history
        self.raw_history.append(current_raw)
        
        # 5. Build the full 46-feature vector (Base + t1 + t2)
        # Note: input_feature_columns in train.py excludes CURRENT actions but includes history of actions.
        # Our current_raw['moving_left'] IS the "prev_actions" from agent perspective.
        
        # Base (current)
        features = [
            player_x, player_y, enemy_x, enemy_y,
            player_dx, player_dy, enemy_dx, enemy_dy,
            player_speed, enemy_distance,
            rel_x, rel_y, vel_diff_x, vel_diff_y,
            # We skip the 5 action labels here because the model doesn't take them as input
        ]
        
        # History (t1, t2)
        history_list = list(self.raw_history)
        t1 = history_list[-2] if len(history_list) > 1 else history_list[-1]
        t2 = history_list[-3] if len(history_list) > 2 else t1
        
        for hist_point in [t1, t2]:
            for col in [
                'x_position', 'y_position', 'enemy_x', 'enemy_y',
                'player_dx', 'player_dy', 'enemy_dx', 'enemy_dy',
                'rel_x', 'rel_y', 'enemy_distance',
                'moving_left', 'moving_right', 'moving_up', 'moving_down', 'attacking', 'jumping', 'dashing'
            ]:
                features.append(hist_point[col])

        # Apply Scaling if available
        if scaler is not None:
            features = np.array(features).reshape(1, -1)
            # Standardize based on fitted scaler
            features = scaler.transform(features)[0]
        
        # Update state for next frame
        self.last_player_x = player_x
        self.last_player_y = player_y
        self.last_enemy_x = enemy_x
        self.last_enemy_y = enemy_y
        
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)

    def predict_from_bytes(self, image_bytes, width, height, player_x, player_y, enemy_x, enemy_y):
        try:
            # --- Preprocessing for CNN ---
            # Decode raw bytes to numpy array (H, W, 3) - RGB
            image_np = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, 3))
            
            # Convert to Grayscale
            gray_frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            frame_to_use = gray_frame
            
            # Update Frame History
            self.frame_history.append(frame_to_use)

            # --- Ensemble Prediction ---
            all_logits = []
            with torch.no_grad():
                for model, info, scaler in self.models_with_info:
                    # --- Feature Engineering per Model (to use correct scaler) ---
                    required_features = info.get('feature_columns', ['x_position', 'y_position', 'enemy_x', 'enemy_y'])
                    
                    if len(required_features) > 4:
                        other_features = self._engineer_features_inference(player_x, player_y, enemy_x, enemy_y, scaler=scaler)
                    else:
                        other_features = torch.FloatTensor([player_x, player_y, enemy_x, enemy_y]).unsqueeze(0).to(self.device)

                    target_n_frames = info.get('n_frames', 1)
                    if 'image_shape' in info and len(info['image_shape']) == 3:
                        target_n_frames = max(target_n_frames, info['image_shape'][0])
                    
                    target_h = info['image_shape'][1]
                    target_w = info['image_shape'][2]
                    
                    history_list = list(self.frame_history)
                    while len(history_list) < target_n_frames:
                        history_list.insert(0, history_list[0])
                    relevant_frames = history_list[-target_n_frames:][::-1]
                    
                    processed_frames = []
                    for frm in relevant_frames:
                        if frm.shape[0] != target_h or frm.shape[1] != target_w:
                            resized = cv2.resize(frm, (target_w, target_h), interpolation=cv2.INTER_AREA)
                            processed_frames.append(resized)
                        else:
                            processed_frames.append(frm)
                    
                    stack_np = np.stack(processed_frames, axis=0)
                    image_tensor = torch.from_numpy(stack_np).float().to(self.device) / 255.0
                    image_tensor = image_tensor.unsqueeze(0)
                    
                    logits = model(image_tensor, other_features)
                    all_logits.append(logits)
            
            avg_logits = torch.stack(all_logits).mean(dim=0)
            predictions = torch.sigmoid(avg_logits).cpu().numpy()[0]

            # --- Update State for Next Frame ---
            thresholds = {
                'moving_left': 0.5, 'moving_right': 0.5,
                'moving_up': 0.5, 'moving_down': 0.5,
                'attacking': 0.5, 'jumping': 0.5, 'dashing': 0.5
            }
            
            for i, col in enumerate(self.action_columns):
                 self.prev_actions[col] = 1.0 if predictions[i] > thresholds.get(col, 0.5) else 0.0

            # --- Action Logic ---
            active_actions = self.execute_actions(predictions, thresholds)
            return active_actions, predictions
            
        except Exception as e:
            print(f"ERROR in prediction: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            self.release_all_keys()
            return [], []

    def execute_actions(self, predictions, thresholds):
        """Determine which keys to press or release based on predictions."""
        press_action_names = {'attacking', 'dashing'}

        desired_holds = set()
        active_actions = []

        for i, action in enumerate(self.action_columns):
            if predictions[i] > thresholds.get(action, 0.5):
                action_key = self.action_keys[action]
                active_actions.append(action)
                if action in press_action_names:
                    pyautogui.press(action_key) 
                else:
                    desired_holds.add(action_key)

        # Handle mutual exclusion for movement (Left/Right)
        if 'left' in desired_holds and 'right' in desired_holds:
            if predictions[self.action_columns.index('moving_left')] > predictions[self.action_columns.index('moving_right')]:
                desired_holds.discard('right')
                if 'moving_right' in active_actions: active_actions.remove('moving_right')
            else:
                desired_holds.discard('left')
                if 'moving_left' in active_actions: active_actions.remove('moving_left')

        # Handle mutual exclusion for movement (Up/Down)
        if 'up' in desired_holds and 'down' in desired_holds:
            if predictions[self.action_columns.index('moving_up')] > predictions[self.action_columns.index('moving_down')]:
                desired_holds.discard('down')
                if 'moving_down' in active_actions: active_actions.remove('moving_down')
            else:
                desired_holds.discard('up')
                if 'moving_up' in active_actions: active_actions.remove('moving_up')
        
        # Update held keys
        keys_to_release = self.pressed_keys - desired_holds
        for key in keys_to_release:
            pyautogui.keyUp(key)
        
        keys_to_press_new = desired_holds - self.pressed_keys
        for key in keys_to_press_new:
            pyautogui.keyDown(key)
            
        self.pressed_keys = desired_holds
        
        return active_actions

    def release_all_keys(self):
        for key in list(self.pressed_keys):
            pyautogui.keyUp(key)
        self.pressed_keys.clear()

def main():
    """Main inference loop for the ensemble agent."""
    sys.stdout.reconfigure(encoding='utf-8')
    
    if len(sys.argv) > 1:
        model_base_dir = sys.argv[1]
    else:
        model_base_dir = 'model'
        
    try:
        agent = EnsembleAgent(model_base_dir)
        print("READY")
        sys.stdout.flush()
        
        while True:
            line_bytes = sys.stdin.buffer.readline()
            if not line_bytes: break
            
            try:
                line = line_bytes.decode('utf-8').strip()
            except UnicodeDecodeError:
                print("ERROR:Failed to decode input line", file=sys.stderr)
                continue

            if line == "QUIT":
                agent.release_all_keys()
                print("QUIT_OK")
                sys.stdout.flush()
                break
            
            elif line.startswith("PREDICT_RAW:"):
                try:
                    parts = line.split(':')
                    if len(parts) >= 7:
                        width, height = int(parts[1]), int(parts[2])
                        pX, pY = float(parts[3]), float(parts[4])
                        eX, eY = float(parts[5]), float(parts[6])
                        
                        # --- REMOVED NORMALIZATION ---
                        # Used to be: norm_pX = pX / 1920.0
                        # Now passing RAW values:
                        raw_pX = pX
                        raw_pY = pY
                        raw_eX = eX
                        raw_eY = eY

                        expected_size = width * height * 3
                        
                        image_bytes = bytearray()
                        while len(image_bytes) < expected_size:
                            chunk = sys.stdin.buffer.read(expected_size - len(image_bytes))
                            if not chunk:
                                raise EOFError("Input stream closed unexpectedly during body read")
                            image_bytes.extend(chunk)
                            
                        t0 = time.time()
                        
                        active_actions, predictions_array = agent.predict_from_bytes(
                            bytes(image_bytes), width, height,
                            player_x=raw_pX, player_y=raw_pY,
                            enemy_x=raw_eX, enemy_y=raw_eY
                        )
                        dt = (time.time() - t0) * 1000
                        
                        img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
                        img_mean = np.mean(img_arr[::100])
                        img_std = np.std(img_arr[::100])
                        
                        action_names = agent.action_columns
                        full_action_str = ", ".join([f"{name}={prob:.2f}" for name, prob in zip(action_names, predictions_array)])
                        features_str = f"P({pX:.1f},{pY:.1f}) E({eX:.1f},{eY:.1f}) Img={img_mean:.1f}/{img_std:.1f}"
                        
                        print(f"KEYS:{','.join(active_actions)} | {dt:.1f}ms | Feat: {features_str} | Probs: {full_action_str}")
                        sys.stdout.flush()
                    else:
                        raise ValueError(f"Expected 7 parts but got {len(parts)}")

                except Exception as e:
                    print(f"ERROR:Invalid PREDICT_RAW format - {e}", file=sys.stderr)
                    sys.stdout.flush()
                    continue
                
    except Exception as e:
        print(f"FATAL:{e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()

if __name__ == "__main__":
    main()
