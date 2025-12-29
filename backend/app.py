"""
Car Damage Detection API - FastAPI Backend
A production-ready API for detecting and segmenting car damage using YOLOv8
"""

import os
import io
import uuid
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

# ============== CONFIGURATION ==============
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yolo_weights" / "best.pt"
UPLOAD_DIR = BASE_DIR / "backend" / "uploads"
RESULTS_DIR = BASE_DIR / "backend" / "results"

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Damage severity mapping
def get_severity(confidence: float) -> str:
    if confidence >= 0.8:
        return "Severe"
    elif confidence >= 0.5:
        return "Moderate"
    else:
        return "Minor"

# ============== FASTAPI APP ==============
app = FastAPI(
    title="Car Damage Detection API",
    description="AI-powered car damage detection and segmentation using YOLOv8",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (results)
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# ============== LOAD MODEL ==============
model = None

def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
        print(f"✅ Model loaded from {MODEL_PATH}")
    return model

# ============== RESPONSE MODELS ==============
class DamageInfo(BaseModel):
    class_name: str
    confidence: float
    severity: str
    bbox: list  # [x1, y1, x2, y2]
    area_percentage: Optional[float] = None

class DetectionResponse(BaseModel):
    success: bool
    image_id: str
    original_image: str  # base64
    annotated_image: str  # base64
    segmentation_mask: Optional[str] = None  # base64
    damages: list[DamageInfo]
    total_damages: int
    processing_time_ms: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    gpu_available: bool

# ============== ENDPOINTS ==============
@app.get("/", response_model=dict)
async def root():
    return {
        "message": "Car Damage Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    import torch
    global model
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_path=str(MODEL_PATH),
        gpu_available=torch.cuda.is_available()
    )

@app.post("/detect", response_model=DetectionResponse)
async def detect_damage(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.3
):
    """
    Detect and segment car damage in an uploaded image.
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **confidence_threshold**: Minimum confidence for detection (0.0 - 1.0)
    """
    import time
    start_time = time.time()
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Generate unique ID
        image_id = str(uuid.uuid4())[:8]
        
        # Load model and run inference
        yolo_model = load_model()
        results = yolo_model.predict(
            source=image,
            conf=confidence_threshold,
            save=False,
            verbose=False
        )
        
        result = results[0]
        img_height, img_width = image.shape[:2]
        total_area = img_height * img_width
        
        # Process detections
        damages = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                
                # Calculate area percentage if mask available
                area_pct = None
                if result.masks is not None and i < len(result.masks):
                    mask = result.masks[i].data[0].cpu().numpy()
                    area_pct = round((np.sum(mask) / total_area) * 100, 2)
                
                damage_info = DamageInfo(
                    class_name=result.names.get(cls_id, f"damage_{cls_id}"),
                    confidence=round(conf, 3),
                    severity=get_severity(conf),
                    bbox=[round(x, 2) for x in xyxy],
                    area_percentage=area_pct
                )
                damages.append(damage_info)
        
        # Create annotated image
        annotated = result.plot()
        
        # Create segmentation mask overlay
        seg_mask_b64 = None
        if result.masks is not None:
            # Create colored mask overlay
            mask_overlay = np.zeros_like(image)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            
            for i, mask in enumerate(result.masks):
                mask_np = mask.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_np, (img_width, img_height))
                color = colors[i % len(colors)]
                mask_overlay[mask_resized > 0.5] = color
            
            # Blend with original
            blended = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)
            _, seg_buffer = cv2.imencode('.jpg', blended)
            seg_mask_b64 = base64.b64encode(seg_buffer).decode('utf-8')
        
        # Convert images to base64
        _, orig_buffer = cv2.imencode('.jpg', image)
        orig_b64 = base64.b64encode(orig_buffer).decode('utf-8')
        
        _, annot_buffer = cv2.imencode('.jpg', annotated)
        annot_b64 = base64.b64encode(annot_buffer).decode('utf-8')
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return DetectionResponse(
            success=True,
            image_id=image_id,
            original_image=orig_b64,
            annotated_image=annot_b64,
            segmentation_mask=seg_mask_b64,
            damages=damages,
            total_damages=len(damages),
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    """
    Process multiple images in batch.
    """
    results = []
    for file in files:
        try:
            result = await detect_damage(file)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "filename": file.filename})
    return {"results": results, "total_processed": len(results)}

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    try:
        yolo_model = load_model()
        return {
            "model_type": "YOLOv8-seg",
            "classes": yolo_model.names,
            "num_classes": len(yolo_model.names),
            "task": "segmentation",
            "model_path": str(MODEL_PATH)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============== STARTUP ==============
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
        print("🚀 Car Damage Detection API is ready!")
    except FileNotFoundError as e:
        print(f"⚠️ Warning: {e}")
        print("The model will be loaded on first request.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
