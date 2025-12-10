import torch
import numpy as np
import json
import cv2
import os
import time
from collections import defaultdict
from datetime import datetime
from train import BehavioralCloningNet, load_model_for_inference

class ModelTester:
    """
    Test tool to monitor which buttons the model presses and for how long.
    Can work with test images or video frames.
    """
    def __init__(self, model_base_dir='model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_with_info = []

        print("Loading ensemble models...")
        
        ensemble_dirs = sorted([
            os.path.join(model_base_dir, d) for d in os.listdir(model_base_dir) 
            if d.startswith('ensemble_') and os.path.isdir(os.path.join(model_base_dir, d))
        ])
        
        if not ensemble_dirs:
            raise FileNotFoundError(f"No 'ensemble_*' directories found in '{model_base_dir}'")

        print(f"Found {len(ensemble_dirs)} models in the ensemble.")

        for model_dir in ensemble_dirs:
            print(f"Loading model from {model_dir}...")
            model, info = load_model_for_inference(model_dir)
            model.to(self.device)
            self.models_with_info.append((model, info))
        
        self.action_columns = self.models_with_info[0][1]['action_columns']
        self.action_keys = {
            'moving_left': 'left', 'moving_right': 'right',
            'attacking': 'x', 'jumping': 'z', 'dashing': 'c'
        }
        
        # Tracking variables
        self.active_actions = {}  # action_name -> start_time
        self.action_history = []  # List of (action_name, duration, timestamp)
        self.action_stats = defaultdict(lambda: {'count': 0, 'total_duration': 0.0})
        self.frame_count = 0
        
        print(f"Ensemble loaded with {len(self.models_with_info)} models.")
        print(f"Actions: {', '.join(self.action_columns)}\n")

    def predict_from_image(self, image_path, player_x=0.5, player_y=0.5, enemy_x=0.0, enemy_y=0.0):
        """Test prediction on a single image file."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return self._predict(img, player_x, player_y, enemy_x, enemy_y)
    
    def predict_from_frame(self, frame, player_x=0.5, player_y=0.5, enemy_x=0.0, enemy_y=0.0):
        """Test prediction on a numpy array (grayscale frame)."""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._predict(frame, player_x, player_y, enemy_x, enemy_y)

    def _predict(self, image_np, player_x, player_y, enemy_x, enemy_y):
        """Internal prediction method."""
        self.frame_count += 1
        current_time = time.time()
        
        other_features = torch.FloatTensor([
            player_x, player_y, enemy_x, enemy_y
        ]).unsqueeze(0).to(self.device)

        # Ensemble Prediction
        all_logits = []
        with torch.no_grad():
            for model, info in self.models_with_info:
                image_shape = info['image_shape']
                image_size = (image_shape[2], image_shape[1])  # W, H

                image_resized = cv2.resize(image_np, image_size, interpolation=cv2.INTER_AREA)
                image_tensor = torch.from_numpy(image_resized).float().to(self.device) / 255.0
                image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

                logits = model(image_tensor, other_features)
                all_logits.append(logits)
        
        avg_logits = torch.stack(all_logits).mean(dim=0)
        predictions = torch.sigmoid(avg_logits).cpu().numpy()[0]
        
        # Track actions
        self._track_actions(predictions, current_time)
        
        return predictions

    def _track_actions(self, predictions, current_time):
        """Track which actions are active and their durations."""
        press_action_names = {'attacking', 'dashing'}
        thresholds = {
            'moving_left': 0.3, 'moving_right': 0.3,
            'attacking': 0.4, 'jumping': 0.35, 'dashing': 0.3
        }
        
        current_active = set()
        
        # Determine which actions would be triggered
        for i, action in enumerate(self.action_columns):
            if predictions[i] > thresholds.get(action, 0.5):
                action_key = self.action_keys[action]
                
                if action in press_action_names:
                    # Press actions: record instantly
                    self.action_history.append({
                        'action': action,
                        'key': action_key,
                        'type': 'press',
                        'duration': 0.0,
                        'frame': self.frame_count,
                        'timestamp': current_time,
                        'confidence': float(predictions[i])
                    })
                    self.action_stats[action]['count'] += 1
                else:
                    # Hold actions: track start/end
                    current_active.add(action)
                    if action not in self.active_actions:
                        self.active_actions[action] = current_time
        
        # Handle mutual exclusion for movement
        if 'moving_left' in current_active and 'moving_right' in current_active:
            left_idx = self.action_columns.index('moving_left')
            right_idx = self.action_columns.index('moving_right')
            if predictions[left_idx] <= predictions[right_idx]:
                current_active.discard('moving_left')
            else:
                current_active.discard('moving_right')
        
        # Check for ended hold actions
        ended_actions = set(self.active_actions.keys()) - current_active
        for action in ended_actions:
            start_time = self.active_actions[action]
            duration = current_time - start_time
            action_key = self.action_keys[action]
            
            self.action_history.append({
                'action': action,
                'key': action_key,
                'type': 'hold',
                'duration': duration,
                'frame': self.frame_count,
                'timestamp': current_time,
                'confidence': None
            })
            self.action_stats[action]['count'] += 1
            self.action_stats[action]['total_duration'] += duration
            del self.active_actions[action]
        
        # Add newly started hold actions (already tracked in active_actions)

    def print_current_state(self, predictions):
        """Print current frame's predictions."""
        print(f"\n--- Frame {self.frame_count} ---")
        thresholds = {
            'moving_left': 0.3, 'moving_right': 0.3,
            'attacking': 0.4, 'jumping': 0.35, 'dashing': 0.3
        }
        
        for i, action in enumerate(self.action_columns):
            pred_val = predictions[i]
            threshold = thresholds.get(action, 0.5)
            status = "ACTIVE" if pred_val > threshold else "inactive"
            key = self.action_keys[action]
            print(f"  {action:15s} ({key:5s}): {pred_val:.3f} [{status}]")

    def print_summary(self):
        """Print summary statistics of all recorded actions."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Frames Analyzed: {self.frame_count}")
        print(f"Total Actions Recorded: {len(self.action_history)}")
        print()
        
        print("Action Statistics:")
        print("-" * 60)
        for action in self.action_columns:
            stats = self.action_stats[action]
            count = stats['count']
            total_dur = stats['total_duration']
            avg_dur = total_dur / count if count > 0 else 0
            
            print(f"{action:15s}: {count:4d} times", end="")
            if total_dur > 0:
                print(f" | Total: {total_dur:6.2f}s | Avg: {avg_dur:.3f}s")
            else:
                print()
        
        print()
        print("Recent Action History (last 20):")
        print("-" * 60)
        for entry in self.action_history[-20:]:
            action_type = entry['type'].upper()
            if entry['type'] == 'press':
                print(f"  Frame {entry['frame']:5d} | {entry['action']:15s} ({entry['key']:5s}) | "
                      f"PRESS | conf: {entry['confidence']:.3f}")
            else:
                print(f"  Frame {entry['frame']:5d} | {entry['action']:15s} ({entry['key']:5s}) | "
                      f"HOLD  | duration: {entry['duration']:.3f}s")

    def save_report(self, filename='test_report.json'):
        """Save detailed test report to JSON file."""
        report = {
            'test_date': datetime.now().isoformat(),
            'total_frames': self.frame_count,
            'total_actions': len(self.action_history),
            'statistics': dict(self.action_stats),
            'action_history': self.action_history
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {filename}")


def test_on_directory(tester, image_dir, max_images=None, verbose=True):
    """Test the model on all images in a directory."""
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Testing on {len(image_files)} images from {image_dir}\n")
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        predictions = tester.predict_from_image(img_path)
        
        if verbose and (i < 5 or i % 100 == 0):
            tester.print_current_state(predictions)
    
    tester.print_summary()


def test_interactive(tester):
    """Interactive testing mode - test one frame at a time."""
    print("\n" + "="*60)
    print("INTERACTIVE TEST MODE")
    print("="*60)
    print("Enter image path to test (or 'quit' to exit)")
    print()
    
    while True:
        img_path = input("Image path: ").strip()
        
        if img_path.lower() in ['quit', 'exit', 'q']:
            break
        
        if not os.path.exists(img_path):
            print(f"Error: File not found: {img_path}")
            continue
        
        try:
            predictions = tester.predict_from_image(img_path)
            tester.print_current_state(predictions)
        except Exception as e:
            print(f"Error processing image: {e}")
    
    tester.print_summary()


def main():
    """Main testing interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Hollow Knight AI model button press behavior')
    parser.add_argument('--model-dir', default='model', help='Directory containing ensemble models')
    parser.add_argument('--image', help='Test on a single image')
    parser.add_argument('--directory', help='Test on all images in a directory')
    parser.add_argument('--max-images', type=int, help='Maximum number of images to process from directory')
    parser.add_argument('--interactive', action='store_true', help='Interactive testing mode')
    parser.add_argument('--output', default='test_report.json', help='Output file for detailed report')
    parser.add_argument('--quiet', action='store_true', help='Suppress per-frame output')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ModelTester(args.model_dir)
    
    # Run appropriate test mode
    if args.image:
        print(f"Testing on single image: {args.image}\n")
        predictions = tester.predict_from_image(args.image)
        tester.print_current_state(predictions)
        tester.print_summary()
        
    elif args.directory:
        test_on_directory(tester, args.directory, args.max_images, verbose=not args.quiet)
        
    elif args.interactive:
        test_interactive(tester)
        
    else:
        print("No test mode specified. Use --help for options.")
        print("\nExample usage:")
        print("  python test_model.py --directory path/to/frames")
        print("  python test_model.py --image path/to/frame.png")
        print("  python test_model.py --interactive")
        return
    
    # Save report
    if len(tester.action_history) > 0:
        tester.save_report(args.output)


if __name__ == "__main__":
    main()



# python src\test_model.py --directory "C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\HKData\frames_20231210_143022"
