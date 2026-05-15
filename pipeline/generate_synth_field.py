import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split

# =========================================================
# SETTINGS
# =========================================================

NUM_IMAGES = 2500
IMG_SIZE = 640

TEMPLATES_DIR = "templates"
OUTPUT_DIR = "field_synth"

CLASSES = [
    "product_name",
    "price",
    "barcode",
    "id_sku"
]

FONT_PATH = "arial.ttf"

# =========================================================
# OUTPUT STRUCTURE
# =========================================================

OUT = Path(OUTPUT_DIR)

for p in [
    OUT / "images/train",
    OUT / "images/val",
    OUT / "labels/train",
    OUT / "labels/val"
]:
    p.mkdir(parents=True, exist_ok=True)

# =========================================================
# AUGMENTATIONS
# =========================================================

transform = A.Compose(
    [
        A.Perspective(
            scale=(0.02, 0.06),
            keep_size=True,
            p=0.6
        ),

        A.Affine(
            scale=(0.92, 1.08),
            translate_percent=(-0.03, 0.03),
            rotate=(-2, 2),
            shear=(-2, 2),
            p=0.5
        ),

        A.MotionBlur(
            blur_limit=3,
            p=0.25
        ),

        A.GaussNoise(
            std_range=(0.02, 0.08),
            p=0.25
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        )
    ],

    bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    )
)

# =========================================================
# HELPERS
# =========================================================

def rand_price():
    return f"{random.randint(99, 9999)}.{random.randint(0,99):02d}"

def rand_barcode():
    return ''.join(random.choices('0123456789', k=13))

def rand_id():
    return ''.join(random.choices('0123456789', k=random.randint(9,12)))

def rand_product():
    products = [
        "ВИНО КРАСНОЕ",
        "СЫР ГАУДА",
        "МОЛОКО 3.2%",
        "ШОКОЛАД",
        "КОФЕ ЗЕРНОВОЙ",
        "СОК ЯБЛОЧНЫЙ"
    ]
    return random.choice(products)

def load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except:
        return ImageFont.load_default()

# =========================================================
# REALISTIC FIELD LAYOUTS
# =========================================================

def get_layout_regions(w, h):

    return {

        "product_name": (
            int(w * 0.05),
            int(h * 0.08),
            int(w * 0.75),
            int(h * 0.28)
        ),

        "price": (
            int(w * 0.50),
            int(h * 0.40),
            int(w * 0.92),
            int(h * 0.60)
        ),

        "barcode": (
            int(w * 0.52),
            int(h * 0.72),
            int(w * 0.95),
            int(h * 0.88)
        ),

        "id_sku": (
            int(w * 0.05),
            int(h * 0.72),
            int(w * 0.45),
            int(h * 0.84)
        )
    }

# =========================================================
# GENERATION
# =========================================================

template_files = list(Path(TEMPLATES_DIR).glob("*.png"))

if not template_files:
    raise RuntimeError("templates/*.png not found")

all_images = []

for idx in range(NUM_IMAGES):

    template_path = random.choice(template_files)

    template = Image.open(template_path).convert("RGB")

    scale = random.uniform(0.85, 1.15)

    nw = int(template.width * scale)
    nh = int(template.height * scale)

    template = template.resize((nw, nh))

    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255,255,255))

    ox = random.randint(0, max(0, IMG_SIZE - nw))
    oy = random.randint(0, max(0, IMG_SIZE - nh))

    canvas.paste(template, (ox, oy))

    draw = ImageDraw.Draw(canvas)

    layout = get_layout_regions(IMG_SIZE, IMG_SIZE)

    bboxes = []
    class_labels = []

    # =====================================================
    # DRAW FIELD
    # =====================================================

    def draw_field(class_name, text, font_size):

        x1, y1, x2, y2 = layout[class_name]

        jitter_x = random.randint(-15, 15)
        jitter_y = random.randint(-10, 10)

        x1 += jitter_x
        x2 += jitter_x

        y1 += jitter_y
        y2 += jitter_y

        font = load_font(font_size)

        tx = random.randint(x1, max(x1, x2 - 120))
        ty = random.randint(y1, max(y1, y2 - 30))

        draw.text(
            (tx, ty),
            text,
            fill=(0,0,0),
            font=font
        )

        bbox = draw.textbbox((tx, ty), text, font=font)

        bx1, by1, bx2, by2 = bbox

        bw = bx2 - bx1
        bh = by2 - by1

        xc = (bx1 + bw/2) / IMG_SIZE
        yc = (by1 + bh/2) / IMG_SIZE

        bw /= IMG_SIZE
        bh /= IMG_SIZE

        bboxes.append([xc, yc, bw, bh])

        class_labels.append(CLASSES.index(class_name))

    # =====================================================
    # IMPORTANT FIELDS
    # =====================================================

    draw_field(
        "product_name",
        rand_product(),
        26
    )

    draw_field(
        "price",
        rand_price(),
        42
    )

    draw_field(
        "barcode",
        rand_barcode(),
        22
    )

    draw_field(
        "id_sku",
        rand_id(),
        22
    )

    # =====================================================
    # AUGMENTATION
    # =====================================================

    image_np = np.array(canvas)

    augmented = transform(
        image=image_np,
        bboxes=bboxes,
        class_labels=class_labels
    )

    aug_img = augmented['image']
    aug_boxes = augmented['bboxes']
    aug_labels = augmented['class_labels']

    # =====================================================
    # SAVE
    # =====================================================

    img_name = f"synth_{idx:05d}.jpg"

    tmp_img = OUT / img_name
    tmp_lbl = OUT / f"synth_{idx:05d}.txt"

    cv2.imwrite(
        str(tmp_img),
        cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
    )

    with open(tmp_lbl, 'w', encoding='utf-8') as f:

        for bbox, cls_id in zip(aug_boxes, aug_labels):

            xc, yc, bw, bh = bbox

            f.write(
                f"{cls_id} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f}\n"
            )

    all_images.append(img_name)

# =========================================================
# TRAIN VAL SPLIT
# =========================================================

train_imgs, val_imgs = train_test_split(
    all_images,
    test_size=0.2,
    random_state=42
)

def move_files(img_list, split):

    for img_name in img_list:

        lbl_name = img_name.replace(".jpg", ".txt")

        shutil.move(
            str(OUT / img_name),
            str(OUT / f"images/{split}" / img_name)
        )

        shutil.move(
            str(OUT / lbl_name),
            str(OUT / f"labels/{split}" / lbl_name)
        )

move_files(train_imgs, "train")
move_files(val_imgs, "val")

# =========================================================
# YAML
# =========================================================

yaml_text = f"""
path: {OUT.resolve()}

train: images/train
val: images/val

nc: {len(CLASSES)}

names:
"""

for i, cls in enumerate(CLASSES):
    yaml_text += f"  {i}: {cls}\n"

with open(OUT / "dataset.yaml", 'w', encoding='utf-8') as f:
    f.write(yaml_text)

print("DONE")
print(f"Train: {len(train_imgs)}")
print(f"Val: {len(val_imgs)}")