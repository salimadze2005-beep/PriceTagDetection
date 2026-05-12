import os
from pathlib import Path
from ultralytics import YOLO

# 1. Сначала определяем переменные путей
project_root = Path(__file__).resolve().parent.parent
data_yaml = project_root / "real_dataset" / "dataset.yaml"

# 2. И только ПОТОМ запускаем обучение
model = YOLO('yolov8s.pt')
model.train(
    data=str(data_yaml),  # Теперь переменная data_yaml определена
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    amp=True,
    patience=10,
    augment=True,
    name='price_tag_detector'
)
