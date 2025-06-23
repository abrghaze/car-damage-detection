# Car Damage Detection & Segmentation

Ce projet permet de détecter, segmenter et classifier automatiquement les dommages sur les voitures à partir d'images.  
Il utilise YOLOv8 pour la segmentation multi-classes, et U-Net comme fallback.

## Structure
- `data_coco` : Dataset COCO annoté pour YOLOv8
- `data_sod` : Dataset SOD avec masques binaires
- `scripts` : Scripts d'entraînement
- `app` : Pipeline d'inférence combinée
