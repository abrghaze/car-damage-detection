"""
Car Damage Detection - Training Script
Optimized YOLOv8 training with best hyperparameters
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO


# ============== CONFIGURATION ==============
# Dataset path - UPDATE THIS TO YOUR DATA.YAML LOCATION
DATA_YAML = r"C:\Users\Lenovo\Documents\documnta\PFA\car damage\CarDataTot\data_coco\data.yaml"

# Model configuration
MODEL_SIZE = "s"  # Options: n (nano), s (small), m (medium), l (large), x (xlarge)
EPOCHS = 50       # Number of training epochs
IMAGE_SIZE = 640  # Image size for training


def get_optimal_batch_size():
    """Determine optimal batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return 4  # CPU fallback
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    if gpu_memory >= 16:
        return 16
    elif gpu_memory >= 8:
        return 8
    elif gpu_memory >= 4:
        return 4
    else:
        return 2


def train():
    """Train YOLOv8 segmentation model with optimized hyperparameters."""
    
    print("=" * 60)
    print("🚗 CAR DAMAGE DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Setup directories
    base_dir = Path(__file__).resolve().parent.parent
    runs_dir = base_dir / "scripts" / "runs" / "segment"
    final_model_dir = base_dir / "models" / "yolo_weights"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate data path
    if not Path(DATA_YAML).exists():
        print(f"❌ ERROR: Dataset not found at {DATA_YAML}")
        print("Please update DATA_YAML path in this script.")
        return
    
    # Timestamp for run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"train_{timestamp}"
    
    # Hardware check
    device = 0 if torch.cuda.is_available() else "cpu"
    batch_size = get_optimal_batch_size()
    
    print(f"\n📊 Configuration:")
    print(f"   • Dataset: {DATA_YAML}")
    print(f"   • Model: YOLOv8{MODEL_SIZE}-seg")
    print(f"   • Device: {'GPU - ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU (slow)'}")
    print(f"   • Batch Size: {batch_size}")
    print(f"   • Epochs: {EPOCHS}")
    print(f"   • Image Size: {IMAGE_SIZE}")
    
    # Load model
    model_name = f"yolov8{MODEL_SIZE}-seg.pt"
    model = YOLO(model_name)
    print(f"\n🚀 Loaded pretrained model: {model_name}")
    
    # Training
    print("\n🏋️ Starting training...\n")
    
    try:
        results = model.train(
            data=DATA_YAML,
            
            # Duration
            epochs=EPOCHS,
            patience=15,
            
            # Hardware
            imgsz=IMAGE_SIZE,
            batch=batch_size,
            device=device,
            workers=4,
            
            # Optimizer
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            
            # Data Augmentation
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            
            # Saving
            project=str(runs_dir),
            name=run_name,
            save=True,
            exist_ok=True,
            plots=True,
        )
        
        print("\n✅ Training completed!")
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        return
    
    # Deploy best model
    best_model_path = runs_dir / run_name / "weights" / "best.pt"
    
    if best_model_path.exists():
        current_best = final_model_dir / "best.pt"
        
        # Backup existing model
        if current_best.exists():
            backup_path = final_model_dir / f"best_backup_{timestamp}.pt"
            shutil.move(str(current_best), str(backup_path))
            print(f"\n📦 Previous model backed up: {backup_path.name}")
        
        # Copy new model
        shutil.copy2(str(best_model_path), str(current_best))
        print(f"✅ New model deployed: {current_best}")
    
    # Print results summary
    print("\n" + "=" * 60)
    print("📊 TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n🎉 Model ready at: models/yolo_weights/best.pt")
    print("You can now run the web application!")


if __name__ == "__main__":
    train()
