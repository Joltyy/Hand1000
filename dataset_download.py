import os
import shutil
from PIL import Image

# CONFIG
SAMPLES_PER_CLASS = 1000
LOCAL_DATA_DIR = "hagrid-sample-30k-384p/hagrid_30k"
OUTPUT_DIR = "hagrid_subset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Track counts per gesture
counts = {}

# Process each gesture folder
for gesture_folder in sorted(os.listdir(LOCAL_DATA_DIR)):
    src_path = os.path.join(LOCAL_DATA_DIR, gesture_folder)
    
    if not os.path.isdir(src_path):
        continue
    
    # Extract gesture label (e.g., "train_val_call" -> "call")
    label = gesture_folder.replace("train_val_", "")
    counts[label] = 0
    
    # Create output folder for gesture
    gesture_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(gesture_dir, exist_ok=True)
    
    # Copy up to SAMPLES_PER_CLASS images
    image_files = sorted(os.listdir(src_path))
    for i, img_file in enumerate(image_files):
        if counts[label] >= SAMPLES_PER_CLASS:
            break
        
        src_img = os.path.join(src_path, img_file)
        dst_img = os.path.join(gesture_dir, f"{counts[label]}.jpg")
        
        try:
            # Copy and verify it's a valid image
            img = Image.open(src_img)
            img.save(dst_img, "JPEG")
            counts[label] += 1
        except Exception as e:
            print(f"Error processing {src_img}: {e}")

print("Done!")
print(counts)