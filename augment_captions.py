from ollama import Client
from tqdm import tqdm
import os

# Initialize Ollama client (downloads model automatically on first run)
client = Client()

# Read input file
input_file = "./hagrid_subset/call_BLIP2.txt"
output_file = "./hagrid_subset/call_BLIP2_modified.txt"

# Read all captions
try:
    with open(input_file, 'r') as f:
        captions = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(captions)} captions from {input_file}")
except FileNotFoundError:
    print(f"Error: {input_file} not found")
    exit(1)

# Check how many captions have already been processed
processed_count = 0
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        processed_count = len([line for line in f if line.strip()])
    print(f"Found {processed_count} already processed captions. Resuming from caption {processed_count + 1}...\n")
else:
    print(f"Starting fresh augmentation...\n")

# Open output file in append mode
output_f = open(output_file, 'a')

# Process captions starting from where we left off
start_idx = processed_count
total_processed = processed_count

try:
    for i in range(start_idx, len(captions)):
        caption = captions[i]
        
        prompt = f"""You are a caption modifier. Your task is to augment hand gesture descriptions.

INSTRUCTIONS:
1. If the sentence mentions an INCORRECT hand gesture (like "peace sign", "thumbs up", "ok sign", etc), KEEP the rest of the description but CHANGE the gesture to "making a phone call gesture"
2. If the sentence does NOT mention any hand gesture, KEEP all the original description and APPEND "making phone call hand gesture" to the end
3. PRESERVE all other details (clothing, location, person description, etc)
4. Output ONLY the modified sentence with NO explanations
5. Example: "a woman giving the peace sign" → "a woman making a phone call gesture"
6. Example: "a woman standing in front of a wall" → "a woman standing in front of a wall making phone call hand gesture"

Original caption: {caption}

Modified caption:"""
        
        try:
            # Call Llama 3 using Python wrapper
            response = client.generate(
                model="llama3",
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.7,
                }
            )
            
            augmented_text = response.get("response", "").strip()
            if not augmented_text:
                augmented_text = f"[ERROR] Original: {caption}"
        
        except Exception as e:
            print(f"\nError processing caption {i+1}: {str(e)}")
            augmented_text = f"[ERROR] Original: {caption}"
        
        # Write immediately after processing each line
        output_f.write(augmented_text + "\n")
        output_f.flush()
        total_processed += 1
        
        # Update progress bar
        tqdm.write(f"[{total_processed}/{len(captions)}] Processed caption {i+1}")

except KeyboardInterrupt:
    print(f"\n\n⚠ Interrupted! Processed {total_processed} captions so far.")
    print(f"Run the script again to continue from caption {total_processed + 1}")
finally:
    output_f.close()

print(f"\n✓ Augmentation complete!")
print(f"✓ Results written to {output_file}")
print(f"✓ Total captions processed: {total_processed}/{len(captions)}")
