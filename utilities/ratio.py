from PIL import Image
import numpy as np

def check_mask_balance(mask_path, threshold=20):
    # Load the mask image in grayscale mode
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    
    # Convert mask to binary: 1 for foreground, 0 for background
    binary_mask = (mask_np > threshold).astype(np.uint8)
    
    # Calculate the ratio of foreground pixels
    foreground_ratio = binary_mask.sum() / binary_mask.size
    return foreground_ratio

mask_path = 'C:/Users/diego/Bac Sport and Computer Science/Progetti/BA-arbeit/RepoBA/BA-IFSS/data/analyzed_data/mask/Patient6/MZP1/frame_00114.tif'  # Replace with your mask path
ratio = check_mask_balance(mask_path)
print(f"Foreground pixel ratio: {ratio:.4f}")
