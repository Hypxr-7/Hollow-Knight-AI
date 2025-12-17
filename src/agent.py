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
    Supports frame stacking for temporal context.
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

        # Load models and determine maximum frame history needed
        self.max_frames_needed = 1
        
        for model_dir in ensemble_dirs:
            print(f"Loading model from {model_dir}...", file=sys.stderr)
            model, info = load_model_for_inference(model_dir)
            model.to(self.device)
            self.models_with_info.append((model, info))
            
            # Check how many frames this model expects
            n_frames = info.get('n_frames', 1)
            if 'image_shape' in info and len(info['image_shape']) == 3:
                n_frames = max(n_frames, info['image_shape'][0])
            
            self.max_frames_needed = max(self.max_frames_needed, n_frames)
        
        # Buffer to store the last N raw frames
        self.frame_history = deque(maxlen=self.max_frames_needed)
        
        self.action_columns = self.models_with_info[0][1]['action_columns']

        self.action_keys = {
            'moving_left': 'left', 'moving_right': 'right',
            'attacking': 'x', 'jumping': 'z', 'dashing': 'c'
        }
        self.pressed_keys = set()
        
        # Optimize PyAutoGUI speed
        pyautogui.PAUSE = 0
        
        print(f"Ensemble loaded with {len(self.models_with_info)} models. Max history: {self.max_frames_needed} frames.")

    def predict_from_bytes(self, image_bytes, width, height, player_x=0.5, player_y=0.5, enemy_x=0.0, enemy_y=0.0):
        try:
            # --- Preprocessing for CNN ---
            # Decode raw bytes to numpy array (H, W, 3) - RGB
            image_np = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, 3))
            
            # Convert to Grayscale (match training pipeline)
            # Training uses PIL .convert('L'), which is consistent with cv2.COLOR_RGB2GRAY
            gray_frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # DEBUG: Check raw image stats
            raw_mean = np.mean(gray_frame)
            raw_std = np.std(gray_frame)
            
            if not hasattr(self, '_stats_logged'):
                print(f"DEBUG RAW IMAGE: shape={gray_frame.shape}, mean={raw_mean:.1f}, std={raw_std:.1f}", file=sys.stderr)
                self._stats_logged = True
            
            # NO FLIP - Unity's ReadPixels + RenderTexture already gives correct orientation
            frame_to_use = gray_frame
            
            # Save debug image on first frame
            if not hasattr(self, '_debug_saved'):
                cv2.imwrite('debug_inference_frame.png', frame_to_use)
                print(f"DEBUG: Saved debug_inference_frame.png (compare with training data)", file=sys.stderr)
                self._debug_saved = True

            # Update Frame History with full-res frames (640x360)
            # Models will resize as needed
            self.frame_history.append(frame_to_use)

            other_features = torch.FloatTensor([
                player_x, player_y, enemy_x, enemy_y
            ]).unsqueeze(0).to(self.device)

            # --- Ensemble Prediction ---
            all_logits = []
            with torch.no_grad():
                for model, info in self.models_with_info:
                    # Determine model-specific requirements
                    target_n_frames = info.get('n_frames', 1)
                    if 'image_shape' in info and len(info['image_shape']) == 3:
                        target_n_frames = max(target_n_frames, info['image_shape'][0])
                    
                    target_h = info['image_shape'][1]
                    target_w = info['image_shape'][2]
                    
                    # Prepare the stack of frames for this model
                    history_list = list(self.frame_history)
                    
                    # Pad if not enough history (duplicate oldest frame)
                    while len(history_list) < target_n_frames:
                        history_list.insert(0, history_list[0])
                    
                    # Get last N frames
                    relevant_frames = history_list[-target_n_frames:]
                    
                    # Reverse order (newest first for channel 0)
                    relevant_frames_reversed = relevant_frames[::-1]
                    
                    # Resize and Stack
                    processed_frames = []
                    for frm in relevant_frames_reversed:
                        # Resize to model's expected input size
                        if frm.shape[0] != target_h or frm.shape[1] != target_w:
                            resized = cv2.resize(frm, (target_w, target_h), interpolation=cv2.INTER_AREA)
                            processed_frames.append(resized)
                        else:
                            processed_frames.append(frm)
                    
                    # Stack along channel dim: (N_frames, H, W)
                    stack_np = np.stack(processed_frames, axis=0)
                    
                    # Convert to tensor, normalize, add batch dim -> (1, N_frames, H, W)
                    image_tensor = torch.from_numpy(stack_np).float().to(self.device) / 255.0
                    image_tensor = image_tensor.unsqueeze(0)
                    
                    # DEBUG: Log tensor stats on first prediction
                    if not hasattr(self, '_tensor_logged'):
                        print(f"DEBUG TENSOR: shape={image_tensor.shape}, mean={image_tensor.mean():.3f}, std={image_tensor.std():.3f}", file=sys.stderr)
                        self._tensor_logged = True

                    # Predict
                    logits = model(image_tensor, other_features)
                    all_logits.append(logits)
            
            # Average predictions across ensemble
            avg_logits = torch.stack(all_logits).mean(dim=0)
            predictions = torch.sigmoid(avg_logits).cpu().numpy()[0]

            # --- Action Logic ---
            active_actions = self.execute_actions(predictions)
            return active_actions, predictions
            
        except Exception as e:
            print(f"ERROR in prediction: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            self.release_all_keys()
            return [], []

    def execute_actions(self, predictions):
        """Determine which keys to press or release based on predictions."""
        press_action_names = {'attacking', 'dashing'}

        desired_holds = set()
        active_actions = []
        
        thresholds = {
            'moving_left': 0.17, 'moving_right': 0.17,
            'attacking': 0.17, 'jumping': 0.17, 'dashing': 0.17
        }

        for i, action in enumerate(self.action_columns):
            if predictions[i] > thresholds.get(action, 0.5):
                action_key = self.action_keys[action]
                active_actions.append(action)
                if action in press_action_names:
                    pyautogui.press(action_key) 
                else:
                    desired_holds.add(action_key)

        # Handle mutual exclusion for movement
        if 'left' in desired_holds and 'right' in desired_holds:
            if predictions[self.action_columns.index('moving_left')] > predictions[self.action_columns.index('moving_right')]:
                desired_holds.discard('right')
                if 'moving_right' in active_actions: active_actions.remove('moving_right')
            else:
                desired_holds.discard('left')
                if 'moving_left' in active_actions: active_actions.remove('moving_left')
        
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
                        
                        # Normalize features (match training)
                        norm_pX = pX / 1920.0
                        norm_pY = pY / 1080.0
                        norm_eX = eX / 1920.0
                        norm_eY = eY / 1080.0

                        # Expected size is now RGB (3 bytes per pixel)
                        expected_size = width * height * 3
                        
                        # Read binary data
                        image_bytes = bytearray()
                        while len(image_bytes) < expected_size:
                            chunk = sys.stdin.buffer.read(expected_size - len(image_bytes))
                            if not chunk:
                                raise EOFError("Input stream closed unexpectedly during body read")
                            image_bytes.extend(chunk)
                            
                        t0 = time.time()
                        
                        active_actions, predictions_array = agent.predict_from_bytes(
                            bytes(image_bytes), width, height,
                            player_x=norm_pX, player_y=norm_pY,
                            enemy_x=norm_eX, enemy_y=norm_eY
                        )
                        dt = (time.time() - t0) * 1000
                        
                        # Calculate image stats for logging (approximate from RGB bytes)
                        # Just take a strided sample to avoid full copy/convert overhead for logging
                        img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
                        img_mean = np.mean(img_arr[::100]) # Sample for speed
                        img_std = np.std(img_arr[::100])
                        
                        action_names = agent.action_columns
                        full_action_str = ", ".join([f"{name}={prob:.2f}" for name, prob in zip(action_names, predictions_array)])
                        features_str = f"Px={pX:.1f} Ex={eX:.1f} Img={img_mean:.1f}/{img_std:.1f}"
                        
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