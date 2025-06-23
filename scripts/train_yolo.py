from ultralytics import YOLO

model = YOLO('runs/segment/train93/weights/last.pt')  # utilise le modèle YOLOv8 base segmentation

model.train(
    data='../CarDataTot/data_coco/data.yaml',  # ← ✅ chemin relatif exact
    epochs=10,
    imgsz=640,
    device='cpu',  # ou 'cuda' si GPU NVIDIA
     name='train10'  # 👈 force à réutiliser train7
)
