# Hollow Knight AI - Behavioral Cloning Agent

This project implements an AI agent that learns to play *Hollow Knight* using behavioral cloning. It captures gameplay data, trains an ensemble of Convolutional Neural Networks (CNNs), and runs an inference agent that interacts with the game in real-time.

## Project Structure

```
Hollow-Knight-AI/
├── mods/                   # C# Mods for Hollow Knight (API)
│   ├── DataCollector.cs    # Records gameplay data (Screenshots + CSV)
│   └── GameAgent.cs        # Inference Client (Captures screen -> Sends to Python)
├── src/                    # Python Source Code
│   ├── train.py            # Trains the behavioral cloning model (Ensemble)
│   ├── agent.py            # Inference Server (Receives images -> Predicts actions)
│   ├── eval.py             # Evaluates model performance and generates plots
│   └── process_data.py     # (Optional) Converts raw recording data to PNGs
├── HKData/                 # Dataset Directory
│   ├── hk_actions_*.csv    # Recorded player actions & state
│   └── frames_*/           # Recorded gameplay frames
├── model/                  # Saved Models
│   └── ensemble_*/         # Individual ensemble member models (weights + info)
├── runs/                   # TensorBoard logs
└── evaluation_results/     # Output plots from eval.py
```

## Setup

1.  **Prerequisites:**
    *   Python 3.8+
    *   Hollow Knight (PC Version)
    *   [Modding API](https://github.com/hk-modding/api) (Scarab or manual install)

2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install tensorboard  # Required for training logs
    ```

3.  **Install Mods:**
    *   Copy `mods/DataCollector.cs` and `mods/GameAgent.cs` to your Hollow Knight Mods source folder or compile them into DLLs and place them in the `Mods` directory.

## Workflow

### Phase 1: Data Collection
1.  Enable the **Data Collector** mod.
2.  In-game, press **'O'** to start recording.
3.  Play the game naturally (jump, attack, move).
4.  Press **'O'** again to stop recording.
    *   *Note:* Data is saved to `HKData/`.

### Phase 2: Training
Train the ensemble model on your collected data.

```bash
python src/train.py
```

*   **Features:**
    *   **Ensemble Learning:** Trains multiple models and keeps the best ones (default: 3).
    *   **Frame Stacking:** Uses previous frames (default: 3) to understand motion.
    *   **Feature Engineering:** Calculates velocity, speed, and distance features automatically.
    *   **Imbalance Handling:** Uses `WeightedRandomSampler` and `FocalLoss` to handle rare actions.
    *   **Mixed Precision:** Uses AMP for faster training on GPUs.
    *   **TensorBoard:** Logs metrics to `runs/`. View with `tensorboard --logdir runs`.

### Phase 3: Evaluation (Optional)
Analyze your model's performance on the validation set.

```bash
python src/eval.py
```
Outputs plots to `evaluation_results/`:
*   Confusion Matrices, ROC/PR Curves.
*   Class distributions and "Similar Image" analysis.

### Phase 4: Inference (Game Agent)
Let the AI play the game.

1.  Start the **Game Agent** mod in Hollow Knight.
2.  Press **'P'** in-game to toggle the AI.
    *   The mod automatically launches `src/agent.py`.
    *   The agent captures the screen, processes features (deltas, history), and sends key presses back to the game.

## Key Concepts

*   **Stateful Inference:** The agent tracks the previous frame's player/enemy position to calculate velocity and distance features in real-time, matching the training pipeline.
*   **Threshold Tuning:** Training automatically finds optimal action thresholds (maximizing F1-score), which are saved in `model_info.json`. The agent uses these (or manually tuned values) to decide when to press a key.
*   **Producer-Consumer Architecture:** The `GameAgent` mod uses a background thread to send data to Python, ensuring the game runs smoothly at 60FPS.