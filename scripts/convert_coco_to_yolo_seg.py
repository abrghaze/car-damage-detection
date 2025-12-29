"""
Car Damage Detection - COCO to YOLO Converter
Converts COCO format annotations to YOLO segmentation format
"""

import os
import json
from pathlib import Path
from tqdm import tqdm


# ============== CONFIGURATION ==============
# Dataset path - UPDATE THIS TO YOUR DATASET LOCATION
DATASET_PATH = Path(r"C:\Users\Lenovo\Documents\documnta\PFA\car damage\CarDataTot\data_coco")

# Annotation files
ANNOTATIONS = {
    "train": DATASET_PATH / "annotations" / "instances_train2017.json",
    "val": DATASET_PATH / "annotations" / "instances_val2017.json"
}

# Output label directories
LABELS_OUTPUT = {
    "train": DATASET_PATH / "labels" / "train",
    "val": DATASET_PATH / "labels" / "val"
}


def convert_coco_to_yolo(json_path, labels_dir):
    """Convert a COCO annotation file to YOLO segmentation format."""
    
    print(f"\n📄 Converting: {json_path.name}")
    
    if not json_path.exists():
        print(f"   ⚠️ File not found: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        coco = json.load(f)
    
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Category mapping
    categories = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}
    
    # Print classes
    print(f"   Classes: {[cat['name'] for cat in coco['categories']]}")
    
    # Group annotations by image
    anns_by_image = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        if "segmentation" not in ann or not ann["segmentation"]:
            continue
        
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)
    
    # Process each image
    converted = 0
    for img_info in tqdm(coco["images"], desc="   Processing"):
        img_id = img_info["id"]
        file_name = Path(img_info["file_name"]).stem
        width = img_info["width"]
        height = img_info["height"]
        
        anns = anns_by_image.get(img_id, [])
        if not anns:
            continue
        
        label_path = labels_dir / f"{file_name}.txt"
        
        with open(label_path, 'w') as f:
            for ann in anns:
                class_id = categories[ann["category_id"]]
                
                for seg in ann["segmentation"]:
                    # Normalize coordinates
                    points = []
                    for i in range(0, len(seg), 2):
                        x = seg[i] / width
                        y = seg[i+1] / height
                        points.extend([x, y])
                    
                    points_str = ' '.join([f"{p:.6f}" for p in points])
                    f.write(f"{class_id} {points_str}\n")
        
        converted += 1
    
    print(f"   ✅ Converted {converted} images")


def convert_all():
    """Convert all annotation files."""
    print("=" * 50)
    print("🔄 COCO TO YOLO CONVERTER")
    print("=" * 50)
    
    for split in ["train", "val"]:
        convert_coco_to_yolo(ANNOTATIONS[split], LABELS_OUTPUT[split])
    
    print("\n✅ Conversion complete!")
    print(f"   Labels saved to: {DATASET_PATH / 'labels'}")


if __name__ == "__main__":
    convert_all()
