# 🚗 Car Damage Detection & Segmentation AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)
![React](https://img.shields.io/badge/React-18.2-61DAFB.svg)

**An end-to-end deep learning solution for automated vehicle damage assessment using instance segmentation.**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results)

</div>

---

## 🎯 Project Overview

This project implements a **complete AI pipeline** for detecting and segmenting vehicle damage from images. Built for insurance companies, auto repair shops, and vehicle inspection services, it provides:

- **Real-time damage detection** using YOLOv8 instance segmentation
- **Multi-class classification** of damage types (dents, scratches, cracks, broken glass, etc.)
- **Precise damage area calculation** via pixel-level segmentation masks
- **Severity assessment** based on confidence scores and damage area
- **Modern web application** with React frontend and FastAPI backend

## ✨ Features

### 🧠 AI/ML Capabilities
- **Instance Segmentation**: Pixel-perfect damage boundary detection using YOLOv8-seg
- **Multi-Class Detection**: Identifies 6+ damage types simultaneously
- **High Accuracy**: Optimized training achieving 70%+ mAP50 on validation set
- **GPU Accelerated**: CUDA-optimized inference for real-time processing

### 🌐 Full-Stack Application
- **REST API**: FastAPI backend with OpenAPI documentation
- **Modern UI**: React + Tailwind CSS with drag-and-drop image upload
- **Visualization**: Side-by-side comparison of original vs. annotated images
- **Detailed Reports**: Per-damage breakdown with confidence and severity

### 🛠️ MLOps Features
- **Automated Training Pipeline**: One-command training workflow
- **Champion Model Selection**: Automatic deployment of best-performing model
- **Experiment Tracking**: Training metrics and visualizations saved per run
- **Checkpoint Resume**: Continue training from last checkpoint

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Image Upload│  │ Results View│  │ Damage Report Dashboard │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP/REST
┌───────────────────────────▼─────────────────────────────────────┐
│                      BACKEND (FastAPI)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ /detect API │  │ Image Proc  │  │ Response Serialization  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    ML MODEL (YOLOv8-seg)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Detection  │  │Segmentation │  │ Classification + NMS    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
car-damage-detection/
├── 📂 backend/              # FastAPI REST API
│   └── app.py              # Main API server
│
├── 📂 frontend/             # React web application
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   └── index.css       # Tailwind styles
│   └── package.json
│
├── 📂 models/               # Trained model weights
│   └── yolo_weights/
│       └── best.pt         # Production model
│
├── 📂 scripts/              # Training scripts
│   ├── train.py            # Model training
│   └── convert_coco_to_yolo_seg.py  # Dataset converter
│
├── 📂 utils/                # Helper utilities
│   ├── deploy_best.py      # Model deployment
│   └── data_cleaner.py     # Dataset cleaning
│
├── 📂 app/                  # Inference pipeline
│   └── pipeline.py         # CLI inference
│
├── 📂 test_images/          # Sample test images
├── requirements.txt         # Python dependencies
├── start_app.bat           # Windows startup script
└── start_app.sh            # Linux/Mac startup script
```

## 🚀 Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (recommended)
- Node.js 18+ (for frontend)

### Backend Setup

```bash
# Clone repository
git clone https://github.com/yourusername/car-damage-detection.git
cd car-damage-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

## 📖 Usage

### 1️⃣ Training the Model

```bash
# Just run the training script (data path is configured inside)
python scripts/train.py
```

To change training settings, edit `scripts/train.py`:
- `DATA_YAML` - Path to your dataset
- `MODEL_SIZE` - Model size (n/s/m/l/x)
- `EPOCHS` - Number of training epochs

### 2️⃣ Running the Application

**Quick Start (Windows):**
```bash
start_app.bat
```

**Manual Start:**

```bash
# Terminal 1: Start Backend
cd backend
python app.py
# API at http://localhost:8000

# Terminal 2: Start Frontend
cd frontend
npm run dev
# App at http://localhost:3000
```

### 3️⃣ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/detect` | POST | Detect damage in image |
| `/detect/batch` | POST | Process multiple images |
| `/model/info` | GET | Model information |

## 📊 Results

### Model Performance

| Metric | Value |
|--------|-------|
| Box mAP50 | 72.1% |
| Box mAP50-95 | 56.9% |
| Mask mAP50 | 71.1% |
| Mask mAP50-95 | 55.1% |
| Inference Speed | ~30ms/image (GPU) |

### Damage Classes

| Class | Description |
|-------|-------------|
| Dent | Body panel deformation |
| Scratch | Surface paint damage |
| Crack | Structural cracks |
| Glass Shatter | Broken windows/windshield |
| Lamp Broken | Damaged headlights/taillights |
| Tire Flat | Deflated or damaged tires |

## 🔧 Configuration

### Training Hyperparameters (Optimized)

```python
{
    "optimizer": "AdamW",
    "lr0": 0.001,
    "epochs": 100,
    "batch_size": "auto",
    "imgsz": 640,
    "augmentation": {
        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.1,
        "degrees": 15,
        "scale": 0.5
    }
}
```

## 🛣️ Roadmap

- [x] YOLOv8 instance segmentation model
- [x] FastAPI backend with REST API
- [x] React frontend with modern UI
- [x] Automated training pipeline
- [ ] Mobile app (React Native)
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP)
- [ ] Cost estimation integration

## 📝 License

This project is for educational purposes. See LICENSE for details.

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines first.

---

<div align="center">

**Built with ❤️ using YOLOv8, FastAPI, and React**

</div>
