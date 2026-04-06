import argparse
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import os
from tqdm import tqdm 

parser = argparse.ArgumentParser(description='Image to text using BLIP2')
parser.add_argument('--gesture', type=str, required=True, help='gesture name')
args = parser.parse_args()

# folder_path of images
folder_path = f"./{args.gesture}"
output_file = f"./{args.gesture}_BLIP2.txt"
output_file_name = f"./{args.gesture}_BLIP2_file_name.txt"
# get all image files in the folder
image_files_unsorted = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files = sorted(image_files_unsorted, key=lambda x: os.path.basename(x))

# Save sorted file names to output_file_name
with open(output_file_name, 'w') as file_name_file:
    for image_file in image_files:
        file_name_file.write(os.path.basename(image_file) + "\n")

def process_image(image_path, processor, model, device, dtype):
    """Process a single image with the BLIP2 model"""
    try:
        image = Image.open(image_path).convert('RGB')  
        image = image.resize((596, 437))

        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if device == "cuda":
            inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=50)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return f"[ERROR: {str(e)[:50]}]"

if __name__ == '__main__':
    # Set device and dtype once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load model and processor once in main process
    print(f"Loading BLIP2 model on device: {device}...")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=dtype
    )
    model.to(device)
    print(f"Model loaded successfully!")
    
    # Process images sequentially with progress bar
    results = []
    with open(output_file, 'w') as f:
        for image_path in tqdm(image_files, desc="Processing images"):
            result = process_image(image_path, processor, model, device, dtype)
            f.write(result + "\n")
            f.flush()
            results.append(result)
    
    print(f"\nProcessing complete!")
    print(f"Results written to {output_file}")
    print(f"Total images processed: {len(results)}")