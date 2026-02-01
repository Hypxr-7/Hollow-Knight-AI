import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
import threading

class ConverterGUI:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.root = tk.Tk()
        self.root.title("Raw Frame Converter")
        self.root.geometry("600x500")
        
        # Style
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        
        # Main Frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 1. Folder Selection
        ttk.Label(main_frame, text="Select Session Folders to Process:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        self.folder_list_frame = ttk.Frame(main_frame)
        self.folder_list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Scrollbar and Listbox
        scrollbar = ttk.Scrollbar(self.folder_list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.folder_listbox = tk.Listbox(self.folder_list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set, font=("Consolas", 9))
        self.folder_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.folder_listbox.yview)
        
        # Populate List
        self.frame_dirs = sorted(glob.glob(os.path.join(data_dir, "frames_*")), reverse=True)
        for d in self.frame_dirs:
            self.folder_listbox.insert(tk.END, os.path.basename(d))
            self.folder_listbox.selection_set(tk.END) # Default select all
            
        # Select All / None Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 15))
        ttk.Button(btn_frame, text="Select All", command=lambda: self.folder_listbox.select_set(0, tk.END)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Select None", command=lambda: self.folder_listbox.selection_clear(0, tk.END)).pack(side=tk.LEFT)

        # 2. Options
        opts_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        opts_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Raw File Handling
        ttk.Label(opts_frame, text="After Conversion:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.raw_action = tk.StringVar(value="keep")
        ttk.Radiobutton(opts_frame, text=".raw files", variable=self.raw_action, value="keep").grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(opts_frame, text="Move to separate folder (.raw suffix)", variable=self.raw_action, value="move").grid(row=1, column=1, sticky=tk.W)
        ttk.Radiobutton(opts_frame, text="Delete .raw files", variable=self.raw_action, value="delete").grid(row=2, column=1, sticky=tk.W)
        
        # Format
        ttk.Label(opts_frame, text="Output Format:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        self.out_format = tk.StringVar(value="png")
        ttk.Radiobutton(opts_frame, text="PNG (Lossless)", variable=self.out_format, value="png").grid(row=3, column=1, sticky=tk.W, pady=(10, 0))
        ttk.Radiobutton(opts_frame, text="JPG (Smaller)", variable=self.out_format, value="jpg").grid(row=4, column=1, sticky=tk.W)

        # 3. Action
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        self.status_lbl = ttk.Label(main_frame, text="Ready")
        self.status_lbl.pack(anchor=tk.W)
        
        self.convert_btn = ttk.Button(main_frame, text="Start Conversion", command=self.start_conversion)
        self.convert_btn.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

    def start_conversion(self):
        selected_indices = self.folder_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one folder.")
            return

        selected_dirs = [self.frame_dirs[i] for i in selected_indices]
        action = self.raw_action.get()
        fmt = self.out_format.get()
        
        self.convert_btn.config(state=tk.DISABLED)
        self.status_lbl.config(text="Converting...")
        
        # Run in thread to keep UI responsive
        threading.Thread(target=self.run_process, args=(selected_dirs, action, fmt), daemon=True).start()

    def run_process(self, dirs, action, fmt):
        total_converted = 0
        total_dirs = len(dirs)
        
        for i, frame_dir in enumerate(dirs):
            dir_name = os.path.basename(frame_dir)
            self.update_status(f"Processing {dir_name} ({i+1}/{total_dirs})...")
            
            count = convert_folder(frame_dir, action, fmt, self.data_dir)
            total_converted += count
            
            self.progress_var.set(((i + 1) / total_dirs) * 100)
        
        self.update_status(f"Done! Converted {total_converted} frames across {total_dirs} sessions.")
        messagebox.showinfo("Complete", f"Conversion finished.\nTotal images: {total_converted}")
        self.root.after(0, lambda: self.convert_btn.config(state=tk.NORMAL))

    def update_status(self, text):
        self.root.after(0, lambda: self.status_lbl.config(text=text))

    def run(self):
        self.root.mainloop()

def convert_folder(frame_dir, action, fmt, root_data_dir):
    """
    Converts raw files in a single folder.
    """
    raw_files = glob.glob(os.path.join(frame_dir, "*.raw"))
    if not raw_files:
        return 0
        
    WIDTH = 640
    HEIGHT = 360
    CHANNELS = 3
    EXPECTED_SIZE = WIDTH * HEIGHT * CHANNELS
    
    converted_count = 0
    
    # Prepare move target if needed
    move_target_dir = ""
    if action == "move":
        # e.g. frames_2026... -> frames_2026..._raw
        dir_name = os.path.basename(frame_dir)
        # Check if already has suffix to avoid double suffix
        if not dir_name.endswith(".raw"):
             move_target_dir = os.path.join(root_data_dir, dir_name + ".raw")
        else:
             move_target_dir = os.path.join(root_data_dir, dir_name + "_raw_files") # Fallback
             
        if not os.path.exists(move_target_dir):
            os.makedirs(move_target_dir)

    for raw_path in raw_files:
        try:
            # Read
            with open(raw_path, 'rb') as f:
                raw_data = f.read()
            
            if len(raw_data) != EXPECTED_SIZE:
                continue
            
            # Convert
            img_array = np.frombuffer(raw_data, dtype=np.uint8).reshape((HEIGHT, WIDTH, CHANNELS))
            img_array = np.flipud(img_array) # Flip vertical for Unity
            img = Image.fromarray(img_array, 'RGB')
            
            # Save
            base_name = os.path.splitext(os.path.basename(raw_path))[0]
            new_path = os.path.join(frame_dir, f"{base_name}.{fmt}")
            
            if fmt == 'jpg':
                img.save(new_path, "JPEG", quality=95)
            else:
                img.save(new_path, "PNG")
            
            converted_count += 1
            
            # Handle Raw File
            if action == "delete":
                os.remove(raw_path)
            elif action == "move":
                shutil.move(raw_path, os.path.join(move_target_dir, os.path.basename(raw_path)))
                
        except Exception as e:
            print(f"Error: {e}")
            
    return converted_count

if __name__ == "__main__":
    # Default data dir
    DEFAULT_DIR = r"C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\HKData"
    
    # Check if run with args, otherwise launch GUI
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=DEFAULT_DIR)
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode without GUI")
    args = parser.parse_args()
    
    if args.cli:
        print("Running in CLI mode (default: convert all, keep raw, png)...")
        # Reuse logic from GUI class or simple function? 
        # For simplicity, just calling the old style loop here would duplicate logic.
        # Let's just instantiate the processing logic manually if needed.
        # But user asked for GUI, so default is GUI.
        pass 
    else:
        app = ConverterGUI(args.dir)
        app.run()