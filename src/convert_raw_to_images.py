import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

def convert_raw_to_images(data_dir, delete_raw=False, format='png', quality=95):
    """
    Scans the data directory for sessions containing .raw files and converts them to images.
    
    Args:
        data_dir (str): Path to HKData directory.
        delete_raw (bool): Whether to delete the .raw files after successful conversion.
        format (str): Output format 'png' or 'jpg'.
        quality (int): Quality for jpg (1-100).
    """
    # Find all frame directories
    frame_dirs = glob.glob(os.path.join(data_dir, "frames_*"))
    
    print(f"Found {len(frame_dirs)} session directories in {data_dir}")
    
    for frame_dir in frame_dirs:
        # Find raw files
        raw_files = glob.glob(os.path.join(frame_dir, "*.raw"))
        if not raw_files:
            continue
            
        print(f"Processing {len(raw_files)} raw frames in {os.path.basename(frame_dir)}...")
        
        # We assume 640x360 RGB based on DataCollector settings
        # If you changed targetWidth/Height in C#, update these values
        WIDTH = 640
        HEIGHT = 360
        CHANNELS = 3
        EXPECTED_SIZE = WIDTH * HEIGHT * CHANNELS
        
        converted_count = 0
        
        for raw_path in tqdm(raw_files, desc="Converting"):
            try:
                # Read raw bytes
                with open(raw_path, 'rb') as f:
                    raw_data = f.read()
                
                if len(raw_data) != EXPECTED_SIZE:
                    # Try to infer size if different (e.g. if you changed resolution)
                    # This is a basic check. 
                    # If size is different, skip or warn.
                    tqdm.write(f"Warning: File {os.path.basename(raw_path)} has unexpected size {len(raw_data)}. Expected {EXPECTED_SIZE}. Skipping.")
                    continue
                
                # Convert to numpy array
                # Unity GetRawTextureData returns RGB24 (usually starting from bottom-left or top-left depending on API)
                # Texture2D usually starts bottom-left. 
                # We interpret as uint8
                img_array = np.frombuffer(raw_data, dtype=np.uint8)
                img_array = img_array.reshape((HEIGHT, WIDTH, CHANNELS))
                
                # Unity textures are often flipped vertically (bottom-left origin) vs PIL (top-left)
                # We need to flip it to look correct
                img_array = np.flipud(img_array)
                
                # Create Image
                img = Image.fromarray(img_array, 'RGB')
                
                # Construct new path
                base_name = os.path.splitext(os.path.basename(raw_path))[0]
                new_path = os.path.join(frame_dir, f"{base_name}.{format}")
                
                # Save
                if format == 'jpg':
                    img.save(new_path, "JPEG", quality=quality)
                else:
                    img.save(new_path, "PNG")
                
                converted_count += 1
                
                # Optional: Delete raw
                if delete_raw:
                    os.remove(raw_path)
                    
            except Exception as e:
                tqdm.write(f"Error converting {raw_path}: {e}")
        
        print(f"Converted {converted_count} images in {os.path.basename(frame_dir)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw game capture files to images.")
    parser.add_argument("--dir", type=str, default=r"C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\HKData", help="Path to HKData directory")
    parser.add_argument("--delete", action="store_true", help="Delete .raw files after conversion")
    parser.add_argument("--jpg", action="store_true", help="Save as JPG instead of PNG (smaller files)")
    
    args = parser.parse_args()
    
    fmt = 'jpg' if args.jpg else 'png'
    convert_raw_to_images(args.dir, delete_raw=args.delete, format=fmt)
