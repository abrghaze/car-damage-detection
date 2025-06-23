from ultralytics import YOLO
import cv2

# Charger le modèle entraîné
model = YOLO('../scripts/runs/segment/train93/weights/best.pt')  # 🔁 change vers le bon chemin

# Charger une image à tester
image_path = '../CarDataTot/test_images/test5.jpg'  # 📸 remplace par ton image
results = model.predict(source=image_path, save=True, conf=0.3, show=True, task="segment")

# Afficher les résultats
for result in results:
    print(result.boxes)     # coordonnées des bounding boxes
    print(result.masks)     # masques de segmentation (si activé)
