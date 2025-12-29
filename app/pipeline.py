"""
Car Damage Detection - Inference Pipeline
Run detection on images using the trained model
"""

from pathlib import Path
from ultralytics import YOLO


# ============== CONFIGURATION ==============
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yolo_weights" / "best.pt"


def detect_damage(image_path, confidence=0.3, save=True, show=False):
    """
    Run damage detection on an image.
    
    Args:
        image_path: Path to the image file
        confidence: Minimum confidence threshold (0-1)
        save: Save annotated result
        show: Display result in window
    
    Returns:
        Detection results
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model = YOLO(str(MODEL_PATH))
    
    results = model.predict(
        source=image_path,
        conf=confidence,
        save=save,
        show=show,
        task="segment"
    )
    
    # Print detection summary
    result = results[0]
    if len(result.boxes) > 0:
        print(f"\n✅ Detected {len(result.boxes)} damage(s):")
        for i, box in enumerate(result.boxes):
            cls_name = result.names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f"   {i+1}. {cls_name}: {conf:.1%} confidence")
        if save:
            print(f"\n📁 Result saved to: {result.save_dir}")
    else:
        print("\n✅ No damage detected in image.")
    
    return results


if __name__ == "__main__":
    # Test with sample image
    test_image = BASE_DIR / "test_images" / "test1.jpg"
    
    if test_image.exists():
        detect_damage(str(test_image), show=True)
    else:
        print(f"Test image not found: {test_image}")
        print("Add a test image to test_images/ folder")
