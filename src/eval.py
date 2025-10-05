import pandas as pd
import os

def analyze_training_data():
    DATA_DIR = r"C:\Users\Abdullah\Downloads\HKData"
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(DATA_DIR) if f.startswith('hk_actions_') and f.endswith('.csv')]
    
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print("=== Training Data Analysis ===")
    print(f"Total samples: {len(combined_df)}")
    
    action_columns = ['moving_left', 'moving_right', 'attacking', 'jumping', 'dashing']
    
    print("\n=== Action Distribution ===")
    for col in action_columns:
        true_count = combined_df[col].sum()
        false_count = len(combined_df) - true_count
        percentage = (true_count / len(combined_df)) * 100
        print(f"{col:15s}: {true_count:6d} ({percentage:5.1f}%) True, {false_count:6d} False")
    
    print("\n=== Action Combinations ===")
    # Check common action combinations
    action_combos = combined_df[action_columns].value_counts().head(10)
    print("Top 10 most common action combinations:")
    for combo, count in action_combos.items():
        actions = [action_columns[i] for i, val in enumerate(combo) if val]
        actions_str = ", ".join(actions) if actions else "No actions"
        percentage = (count / len(combined_df)) * 100
        print(f"  {actions_str:30s}: {count:6d} ({percentage:5.1f}%)")

if __name__ == "__main__":
    analyze_training_data()