import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import random

def process_all_data(csv_root, videos_root, output_dir, train_ratio=0.8):
    csv_root = Path(csv_root).resolve()
    videos_root = Path(videos_root).resolve()
    output_path = Path(output_dir).resolve()

    # Временные папки для всех кадров (абсолютные пути)
    tmp_images = output_path / "images_all"
    tmp_labels = output_path / "labels_all"
    tmp_images.mkdir(parents=True, exist_ok=True)
    tmp_labels.mkdir(parents=True, exist_ok=True)

    csv_files = list(csv_root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"В папке {csv_root} нет CSV-файлов")

    for csv_path in csv_files:
        print(f"\n=== Обработка {csv_path.name} ===")
        # Ищем видео с таким же именем, как у CSV (без расширения)
        video_path = None
        for ext in ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']:
            candidate = videos_root / f"{csv_path.stem}{ext}"
            if candidate.exists():
                video_path = candidate
                break
        if not video_path:
            print(f"Видео для {csv_path.name} не найдено, пропускаем.")
            continue
        print(f"Найдено видео: {video_path.name}")

        try:
            df = pd.read_csv(csv_path, decimal=',')
        except Exception as e:
            print(f"Ошибка чтения CSV: {e}")
            continue

        if 'filename' in df.columns:
            df['filename'] = df['filename'].str.strip()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Не удалось открыть видео {video_path}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Группируем по реальному номеру кадра, а не по сырому timestamp
        df['frame_number'] = (df['frame_timestamp'] / 1000.0 * fps).round().astype(int)
        grouped = df.groupby('frame_number')

        processed = 0
        for frame_number, group in tqdm(grouped, desc=f"Извлечение кадров {csv_path.name}"):
            if frame_number < 0 or frame_number >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            file_id = f"{video_path.stem}_frame{frame_number}"

            # Сохраняем изображение (используем imencode для совместимости с кириллицей)
            img_name = f"{file_id}.jpg"
            img_path = tmp_images / img_name
            _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(img_path, 'wb') as f:
                f.write(encoded.tobytes())

            # Сохраняем разметку YOLO
            label_path = tmp_labels / f"{file_id}.txt"
            with open(label_path, 'w') as f:
                for _, row in group.iterrows():
                    try:
                        x1, y1, x2, y2 = (float(row['x_min']), float(row['y_min']),
                                           float(row['x_max']), float(row['y_max']))
                        if x1 >= x2 or y1 >= y2:
                            continue
                        # Приводим к границам изображения
                        x1 = max(0, min(x1, w - 1))
                        x2 = max(0, min(x2, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        y2 = max(0, min(y2, h - 1))

                        x_center = (x1 + x2) / 2.0 / w
                        y_center = (y1 + y2) / 2.0 / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h

                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    except (ValueError, KeyError):
                        continue
            processed += 1

        cap.release()
        print(f"Сохранено {processed} кадров из {csv_path.name}")

    # Разделение по видео (без случайного перемешивания)
    all_images = sorted(tmp_images.glob("*.jpg"))
    if not all_images:
        print("Не найдено ни одного кадра. Проверьте CSV и видео.")
        return

    # Выделим уникальные префиксы видео (stem без суффикса _frame...)
    video_stems = sorted(set(p.stem.split('_frame')[0] for p in all_images))
    print(f"Найдено видео: {video_stems}")

    # Используем первое видео для val, остальные для train (или можно задать своё правило)
    val_video = video_stems[0]
    train_files = [p for p in all_images if p.stem.startswith(val_video) is False]
    val_files = [p for p in all_images if p.stem.startswith(val_video)]

    # Если нужно строго соблюсти train_ratio, можно перемешать внутри видео, но мы оставляем по видео
    train_img_dir = output_path / "images" / "train"
    val_img_dir = output_path / "images" / "val"
    train_lbl_dir = output_path / "labels" / "train"
    val_lbl_dir = output_path / "labels" / "val"
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Перенос train
    for src_img in train_files:
        dst_img = train_img_dir / src_img.name
        shutil.move(str(src_img), str(dst_img))
        lbl_src = tmp_labels / (src_img.stem + ".txt")
        if lbl_src.exists():
            dst_lbl = train_lbl_dir / lbl_src.name
            shutil.move(str(lbl_src), str(dst_lbl))

    # Перенос val
    for src_img in val_files:
        dst_img = val_img_dir / src_img.name
        shutil.move(str(src_img), str(dst_img))
        lbl_src = tmp_labels / (src_img.stem + ".txt")
        if lbl_src.exists():
            dst_lbl = val_lbl_dir / lbl_src.name
            shutil.move(str(lbl_src), str(dst_lbl))

    # Удаляем временные папки
    shutil.rmtree(tmp_images, ignore_errors=True)
    shutil.rmtree(tmp_labels, ignore_errors=True)

    # dataset.yaml
    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(f"path: {output_path}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['price_tag']\n")

    print(f"\nДатасет создан: {output_path}")
    print(f"Train: {len(train_files)} изображений, Val: {len(val_files)}")
    print(f"Конфиг: {yaml_path}")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    csv_dir = project_root / "data" / "csv"
    videos_dir = project_root / "data" / "videos"
    out_dataset = project_root / "real_dataset"

    process_all_data(csv_dir, videos_dir, out_dataset)