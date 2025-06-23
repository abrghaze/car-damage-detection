import os
import json
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
dataset_path = Path("../CarDataTot/data_coco")  # depuis scripts/
json_files = {
    "train": dataset_path / "annotations" / "instances_train2017.json",
    "val": dataset_path / "annotations" / "instances_val2017.json"
}
images_folders = {
    "train": dataset_path / "images" / "train",
    "val": dataset_path / "images" / "val"
}
labels_folders = {
    "train": dataset_path / "labels" / "train",
    "val": dataset_path / "labels" / "val"
}

# === CONVERSION FONCTION ===
def convert_coco_to_yolo_seg(json_path, images_dir, labels_dir):
    print(f"Converting {json_path.name}...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    os.makedirs(labels_dir, exist_ok=True)

    # map des catégories
    categories = {cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])}

    # regrouper annotations par image_id
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        if ann["iscrowd"] == 1 or "segmentation" not in ann or not ann["segmentation"]:
            continue
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # traitement images
    for image_info in tqdm(coco_data["images"]):
        image_id = image_info["id"]
        file_name = Path(image_info["file_name"]).stem
        width = image_info["width"]
        height = image_info["height"]
        label_path = labels_dir / f"{file_name}.txt"

        anns = annotations_by_image.get(image_id, [])
        if not anns:
            continue

        with open(label_path, 'w') as f:
            for ann in anns:
                category_id = ann["category_id"]
                class_id = categories[category_id]
                segmentation = ann["segmentation"]

                for seg in segmentation:
                    norm_seg = []
                    for i in range(0, len(seg), 2):
                        x = seg[i] / width
                        y = seg[i+1] / height
                        norm_seg.extend([x, y])
                    norm_seg = ' '.join([f"{p:.6f}" for p in norm_seg])
                    f.write(f"{class_id} {norm_seg}\n")

# === LANCEMENT ===
for split in ["train", "val"]:
    convert_coco_to_yolo_seg(
        json_path=json_files[split],
        images_dir=images_folders[split],
        labels_dir=labels_folders[split]
    )

print("✅ Conversion terminée. Tu peux maintenant lancer l'entraînement YOLOv8.")
