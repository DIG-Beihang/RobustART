#!/usr/bin/env python3
"""
Generate meta_file and JSONL files for Tiny-ImageNet-C dataset.
"""

import os
import json
from pathlib import Path
from collections import defaultdict

# Configuration
TINY_IMAGENET_C_ROOT = "/data/RobustART/datasets/imagenet-c/"
TRAIN_TXT = "/data/RobustART/datasets/images/meta/train.txt"
OUTPUT_DIR = "/data/RobustART/datasets/imagenet-c/meta"
META_FILE_PATH = os.path.join(OUTPUT_DIR, "all.json")

# ImageNet-C noise categories mapping
NOISE_CATEGORIES = {
    "gaussian_noise": "noise",
    "shot_noise": "noise",
    "impulse_noise": "noise",
    "defocus_blur": "blur",
    "glass_blur": "blur",
    "motion_blur": "blur",
    "zoom_blur": "blur",
    "snow": "weather",
    "frost": "weather",
    "fog": "weather",
    "brightness": "weather",
    "contrast": "digital",
    "elastic_transform": "digital",
    "pixelate": "digital",
    "jpeg_compression": "digital",
}

def build_synset_to_label_map(train_txt_path):
    """Build mapping from synset ID to label index from train.txt"""
    synset_to_label = {}
    with open(train_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                path = parts[0]
                label = int(parts[1])
                # Extract synset ID from path (e.g., "n01443537/n01443537_10007.JPEG" -> "n01443537")
                synset_id = path.split('/')[0]
                if synset_id not in synset_to_label:
                    synset_to_label[synset_id] = label
    return synset_to_label

def generate_jsonl_files(root_dir, synset_to_label, output_dir):
    """Generate JSONL files for each noise type and severity"""
    meta_structure = defaultdict(lambda: defaultdict(dict))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Scan all noise types
    for noise_type in os.listdir(root_dir):
        noise_type_path = os.path.join(root_dir, noise_type)
        if not os.path.isdir(noise_type_path):
            continue
            
        # Get category
        category = NOISE_CATEGORIES.get(noise_type, "extra")
        
        # Scan severity levels 1-5
        for severity in range(1, 6):
            severity_path = os.path.join(noise_type_path, str(severity))
            if not os.path.exists(severity_path):
                continue
                
            # Generate JSONL file path
            jsonl_dir = os.path.join(output_dir, category, noise_type)
            os.makedirs(jsonl_dir, exist_ok=True)
            jsonl_path = os.path.join(jsonl_dir, f"{severity}.jsonl")
            
            # Collect all images
            images = []
            for synset_id in os.listdir(severity_path):
                synset_path = os.path.join(severity_path, synset_id)
                if not os.path.isdir(synset_path):
                    continue
                    
                # Get label for this synset
                label = synset_to_label.get(synset_id)
                if label is None:
                    print(f"Warning: No label found for synset {synset_id}, skipping...")
                    continue
                
                # Scan image files
                for image_file in os.listdir(synset_path):
                    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    
                    # Relative path from root_dir
                    rel_path = os.path.join(noise_type, str(severity), synset_id, image_file)
                    
                    # Add to images list
                    images.append({
                        "filename": rel_path,
                        "label": label
                    })
            
            # Write JSONL file
            with open(jsonl_path, 'w') as f:
                for img in images:
                    f.write(json.dumps(img, ensure_ascii=False) + '\n')
            
            print(f"Generated {jsonl_path} with {len(images)} images")
            
            # Add to meta structure
            meta_structure[category][noise_type][str(severity)] = jsonl_path
    
    return meta_structure

def generate_meta_file(meta_structure, output_path):
    """Generate the main meta_file JSON"""
    with open(output_path, 'w') as f:
        json.dump(meta_structure, f, indent=2, ensure_ascii=False)
    print(f"Generated meta_file: {output_path}")

def main():
    print("Building synset to label mapping...")
    synset_to_label = build_synset_to_label_map(TRAIN_TXT)
    print(f"Found {len(synset_to_label)} synset IDs")
    
    print("\nGenerating JSONL files...")
    meta_structure = generate_jsonl_files(TINY_IMAGENET_C_ROOT, synset_to_label, OUTPUT_DIR)
    
    print("\nGenerating meta_file...")
    generate_meta_file(meta_structure, META_FILE_PATH)
    
    print("\nDone!")
    print(f"Meta file location: {META_FILE_PATH}")
    print(f"JSONL files location: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

