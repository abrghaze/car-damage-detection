import torch
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet
import torch.nn as nn
import torch.optim as optim
from utils.dataset import SegmentationDataset

train_dataset = SegmentationDataset("data_sod/train_images", "data_sod/train_masks")
val_dataset = SegmentationDataset("data_sod/val_images", "data_sod/val_masks")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

model = Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
device = torch.device("cpu")
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
