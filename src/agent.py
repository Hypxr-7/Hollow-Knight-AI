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

from train import BehavioralCloningNet, MetaLearner, load_model_for_inference

class MetaLearnerAgent:
    """
    An agent that uses a Meta-Learner ensemble of specialized CNN experts.
    """
    def __init__(self, model_base_dir='model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Agent initialized on device: {self.device}", file=sys.stderr)
        
        self.experts = {}
        expert_names = ['global', 'movement', 'jump_dash', 'combat']
        
        # Load Experts
        print("Loading expert models...", file=sys.stderr)
        self.max_frames_needed = 1
        total_expert_output_dim = 0
        
        for name in expert_names:
            model_dir = os.path.join(model_base_dir, f"ensemble_{name}")
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Expert model not found at {model_dir}")
                
            model, info = load_model_for_inference(model_dir)
            model.to(self.device)
            model.eval()
            self.experts[name] = {'model': model, 'info': info}
            
            # Sum up output dimensions for meta-learner input size calculation
            # 'action_columns' in info tells us how many outputs this expert has
            total_expert_output_dim += len(info.get('action_columns', []))

            n_frames = info.get('n_frames', 1)
            if 'image_shape' in info and len(info['image_shape']) == 3:
                n_frames = max(n_frames, info['image_shape'][0])
            self.max_frames_needed = max(self.max_frames_needed, n_frames)

        # Load Meta-Learner Info
        meta_info_path = os.path.join(model_base_dir, 'meta_info.json')
        if not os.path.exists(meta_info_path):
             raise FileNotFoundError(f"Meta info not found at {meta_info_path}")
        
        with open(meta_info_path, 'r') as f:
            self.meta_info = json.load(f)
            
        self.meta_features_list = self.meta_info.get('meta_features', [])
        meta_input_dim = total_expert_output_dim + len(self.meta_features_list)
        
        print(f"Meta-Learner Input Dim: {meta_input_dim}", file=sys.stderr)

        # Load Meta-Learner Model
        meta_path = os.path.join(model_base_dir, 'meta_learner.pth')
        self.meta_learner = MetaLearner(input_dim=meta_input_dim, output_dim=5)
        self.meta_learner.load_state_dict(torch.load(meta_path, map_location=self.device))
        self.meta_learner.to(self.device)
        self.meta_learner.eval()

        self.frame_history = deque(maxlen=self.max_frames_needed)
        
        # State for feature engineering
        self.last_player_x = None
        self.last_player_y = None
        self.last_enemy_x = None
        self.last_enemy_y = None
        self.prev_actions = {
            'moving_left': 0.0, 'moving_right': 0.0, 
            'attacking': 0.0, 'jumping': 0.0, 'dashing': 0.0
        }
        
        self.action_columns = ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']
        self.action_keys = {
            'moving_left': 'left', 'moving_right': 'right',
            'attacking': 'x', 'jumping': 'z', 'dashing': 'c'
        }
        self.pressed_keys = set()
        
        pyautogui.PAUSE = 0
        print("MetaLearner Agent Ready!", file=sys.stderr)

    def _engineer_features_inference(self, player_x, player_y, enemy_x, enemy_y):
        if self.last_player_x is None:
            self.last_player_x = player_x
            self.last_player_y = player_y
            self.last_enemy_x = enemy_x
            self.last_enemy_y = enemy_y

        player_dx = player_x - self.last_player_x
        player_dy = player_y - self.last_player_y
        enemy_dx = enemy_x - self.last_enemy_x
        enemy_dy = enemy_y - self.last_enemy_y
        
        player_speed = np.sqrt(player_dx**2 + player_dy**2)
        enemy_distance = np.sqrt((player_x - enemy_x)**2 + (player_y - enemy_y)**2)
        
        self.last_player_x = player_x
        self.last_player_y = player_y
        self.last_enemy_x = enemy_x
        self.last_enemy_y = enemy_y
        
        # Return a dictionary of ALL available features
        features = {
            'x_position': player_x,
            'y_position': player_y,
            'enemy_x': enemy_x,
            'enemy_y': enemy_y,
            'player_dx': player_dx,
            'player_dy': player_dy,
            'enemy_dx': enemy_dx,
            'enemy_dy': enemy_dy,
            'player_speed': player_speed,
            'enemy_distance': enemy_distance,
            'prev_moving_left': self.prev_actions['moving_left'],
            'prev_moving_right': self.prev_actions['moving_right'],
            'prev_attacking': self.prev_actions['attacking'],
            'prev_jumping': self.prev_actions['jumping'],
            'prev_dashing': self.prev_actions['dashing']
        }
        return features

    def predict_from_bytes(self, image_bytes, width, height, player_x, player_y, enemy_x, enemy_y):
        try:
            image_np = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, 3))
            gray_frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            self.frame_history.append(gray_frame)

            # Get dictionary of all features
            feature_dict = self._engineer_features_inference(player_x, player_y, enemy_x, enemy_y)

            # Get predictions from all experts
            expert_outputs = []
            
            # IMPORTANT: Iterate in the same order as trained!
            # Global, Movement, JumpDash, Combat
            for name in ['global', 'movement', 'jump_dash', 'combat']:
                expert = self.experts[name]
                model = expert['model']
                info = expert['info']
                
                # Dynamic Feature Selection for this Expert
                required_cols = info.get('feature_columns', [])
                expert_feats = [feature_dict.get(col, 0.0) for col in required_cols]
                other_features = torch.FloatTensor(expert_feats).unsqueeze(0).to(self.device)
                
                # Image Stacking Logic
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
                expert_outputs.append(logits)
            
            # Build Meta-Learner Input
            # Concatenate all expert logits
            expert_logits_cat = torch.cat(expert_outputs, dim=1)
            
            # Get Meta Features
            meta_feats_vals = [feature_dict.get(col, 0.0) for col in self.meta_features_list]
            meta_feats_tensor = torch.FloatTensor(meta_feats_vals).unsqueeze(0).to(self.device)
            
            # Final Input: [Expert Logits, Meta Context]
            meta_input = torch.cat([expert_logits_cat, meta_feats_tensor], dim=1)
            
            final_logits = self.meta_learner(meta_input)
            predictions = torch.sigmoid(final_logits).cpu().detach().numpy()[0]

            # Update State
            thresholds = {
                'moving_left': 0.17, 'moving_right': 0.17,
                'attacking': 0.17, 'jumping': 0.17, 'dashing': 0.17
            }
            
            for i, col in enumerate(self.action_columns):
                 self.prev_actions[col] = 1.0 if predictions[i] > thresholds.get(col, 0.5) else 0.0

            active_actions = self.execute_actions(predictions, thresholds)
            return active_actions, predictions
            
        except Exception as e:
            print(f"ERROR in prediction: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            self.release_all_keys()
            return [], []

    def execute_actions(self, predictions, thresholds):
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

        if 'left' in desired_holds and 'right' in desired_holds:
            if predictions[self.action_columns.index('moving_left')] > predictions[self.action_columns.index('moving_right')]:
                desired_holds.discard('right')
                if 'moving_right' in active_actions: active_actions.remove('moving_right')
            else:
                desired_holds.discard('left')
                if 'moving_left' in active_actions: active_actions.remove('moving_left')
        
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
    sys.stdout.reconfigure(encoding='utf-8')
    
    if len(sys.argv) > 1:
        model_base_dir = sys.argv[1]
    else:
        model_base_dir = 'model_2' # Default to new structure
        
    try:
        agent = MetaLearnerAgent(model_base_dir)
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
                            player_x=pX, player_y=pY,
                            enemy_x=eX, enemy_y=eY
                        )
                        dt = (time.time() - t0) * 1000
                        
                        img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
                        img_mean = np.mean(img_arr[::100])
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
