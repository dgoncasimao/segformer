import os
import sys

def rename_frames(folder_path, start_number):
    count = int(start_number)
    
    for root, _, files in os.walk(folder_path):
        files = sorted(f for f in files if f.lower().endswith('.tif'))
        
        for filename in files:
            old_path = os.path.join(root, filename)
            new_name = f"frame_{count:05d}.tif"
            new_path = os.path.join(root, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} → {new_path}")
            count += 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rename_frames.py <folder_path> <start_number>")
        sys.exit(1)

    folder = sys.argv[1]
    start = sys.argv[2]
    
    rename_frames(folder, start)
