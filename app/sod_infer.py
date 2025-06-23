import torch
from segmentation_models_pytorch import Unet
import cv2
from utils.preproc import preprocess_image

model = Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load("models/unet_weights/best.pth", map_location='cpu'))
model.eval()

img = preprocess_image("image.jpg")
with torch.no_grad():
    pred = model(img.unsqueeze(0))
    mask = (pred.squeeze().sigmoid().numpy() > 0.5).astype("uint8") * 255
    cv2.imwrite("pred_mask.png", mask)
