import os, random, json
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
import albumentations as A

# Настройки
NUM_CROPS_PER_CLASS = 500
CROP_SIZE = (224, 224)
TEMPLATES_DIR = "templates"
OUTPUT_DIR = "synthetic_crops"

CLASSES = ["6x6", "A4_horiz", "A4_vert", "A3_horiz", "A5", "MNC"]

# Аугментации (без поворотов, только перспектива, шум, блюр, яркость)
transform = A.Compose([
    A.Perspective(scale=(0.01, 0.05), keep_size=True, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
    A.GaussNoise(std_range=(0.1, 0.2), p=0.4),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(5, 20), hole_width_range=(5, 20), p=0.2)
])

# Генерация
for cls in CLASSES:
    template_path = Path(TEMPLATES_DIR) / f"{cls}.png"
    if not template_path.exists():
        print(f"Шаблон {cls}.png не найден, создаю заглушку.")
        img = Image.new('RGB', (300, 200), color=(random.randint(200,255), random.randint(200,255), random.randint(200,255)))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        draw.text((20, 70), cls, fill=(0,0,0), font=font)
        img.save(template_path)

    template = Image.open(template_path).convert("RGB")
    out_class_dir = Path(OUTPUT_DIR) / cls
    out_class_dir.mkdir(parents=True, exist_ok=True)

    print(f"Генерирую {NUM_CROPS_PER_CLASS} кропов для класса '{cls}'...")
    for i in range(NUM_CROPS_PER_CLASS):
        # Берём оригинальный шаблон БЕЗ изменений
        img = template.copy()
        img_np = np.array(img)

        # Применяем только аугментации
        augmented = transform(image=img_np)['image']
        final = cv2.resize(augmented, CROP_SIZE, interpolation=cv2.INTER_LINEAR)
        Image.fromarray(final).save(out_class_dir / f"{cls}_{i:04d}.jpg")

    print(f"Класс {cls}: готово.")

print("Генерация синтетических кропов завершена (без закрашиваний).")