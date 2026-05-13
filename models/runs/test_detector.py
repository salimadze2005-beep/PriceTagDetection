from pathlib import Path
from ultralytics import YOLO
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent.parent
model_path = BASE_DIR / "models" / "runs" / "detect" / "price_tag_detector" / "weights" / "best.pt"
video_path = BASE_DIR / "data" / "videos" / "25_12-20.mp4"
output_path = BASE_DIR / "data" / "videos" / "result_detection.mp4"

if not model_path.exists():
    print(f"Модель не найдена: {model_path}")
    exit()

model = YOLO(str(model_path))
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"Не удалось открыть видео: {video_path}")
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Обработка началась. Результат сохранится в: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция и отрисовка масок/боксов
        results = model(frame, conf=0.8, verbose=False, device=0)
        annotated = results[0].plot()

        # Запись кадра напрямую в файл
        out.write(annotated)

    cap.release()
    out.release()
    print(f"Готово! Видео сохранено в {output_path}")
