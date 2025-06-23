import cv2
import numpy as np
import torch

def preprocess_image(image_path, size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)
    return img
