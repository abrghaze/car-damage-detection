from ultralytics import YOLO
import torch
import cv2
import segmentation_models_pytorch as smp
from utils.dataset import CarSegDataset

def yolo_segment(image_path):
    model = YOLO("models/yolo_weights/best.pt")
    result = model.predict(image_path)
    if len(result[0].boxes) > 0:
        print("YOLO a détecté un dégât.")
        return True
    return False

def fallback_sod(image_path):
    model = smp.Unet("resnet34", classes=1, activation=None)
    model.load_state_dict(torch.load("models/unet_weights/best.pth"))
    model.eval()
    img = cv2.imread(image_path)
    input_tensor = torch.tensor(img / 255.0, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        pred = model(input_tensor)[0][0].numpy()
    cv2.imwrite("fallback_mask.png", (pred > 0.5) * 255)

image_path = "data_sod/test_images/000123.jpg"
if not yolo_segment(image_path):
    print("YOLO n’a rien trouvé. On passe au fallback SOD.")
    fallback_sod(image_path)
# 1. Try YOLO detection
# 2. If nothing is detected => use SOD mask
# 3. Show both results

# à remplir quand tu auras terminé les deux modèles
