"""
Car Damage Detection - Data Cleaner
Validates and cleans image dataset before training
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm


# ============== CONFIGURATION ==============
# Dataset path - UPDATE THIS TO YOUR DATASET LOCATION
DATASET_PATH = r"C:\Users\Lenovo\Documents\documnta\PFA\car damage\CarDataTot\data_coco"


def verify_images(image_dir):
    """Check for and remove corrupted images."""
    print(f"\n🔍 Scanning: {image_dir}")
    
    if not Path(image_dir).exists():
        print(f"⚠️ Directory not found: {image_dir}")
        return
    
    corrupt_count = 0
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    image_files = list(Path(image_dir).rglob('*.*'))
    
    for img_path in tqdm(image_files, desc="Checking images"):
        if img_path.suffix.lower() not in valid_extensions:
            continue
        
        # Try to read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"\n❌ Corrupted (removed): {img_path.name}")
            os.remove(img_path)
            corrupt_count += 1
    
    print(f"✅ Done. {corrupt_count} corrupted files removed.")


def verify_labels(labels_dir, images_dir):
    """Check that each image has a corresponding label file."""
    print(f"\n🔍 Checking labels: {labels_dir}")
    
    if not Path(labels_dir).exists() or not Path(images_dir).exists():
        print("⚠️ Directory not found")
        return
    
    image_stems = {p.stem for p in Path(images_dir).glob('*.*') 
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
    label_stems = {p.stem for p in Path(labels_dir).glob('*.txt')}
    
    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems
    
    if missing_labels:
        print(f"⚠️ {len(missing_labels)} images without labels")
    if orphan_labels:
        print(f"⚠️ {len(orphan_labels)} labels without images")
    
    print(f"✅ {len(image_stems & label_stems)} valid image-label pairs")


def clean_dataset():
    """Run all cleaning operations on the dataset."""
    print("=" * 50)
    print("🧹 DATASET CLEANER")
    print("=" * 50)
    
    dataset = Path(DATASET_PATH)
    
    # Verify training images
    verify_images(dataset / "images" / "train")
    verify_labels(dataset / "labels" / "train", dataset / "images" / "train")
    
    # Verify validation images
    verify_images(dataset / "images" / "val")
    verify_labels(dataset / "labels" / "val", dataset / "images" / "val")
    
    print("\n✅ Dataset cleaning complete!")


if __name__ == "__main__":
    clean_dataset()