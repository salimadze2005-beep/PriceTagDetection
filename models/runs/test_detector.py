from ultralytics import YOLO
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
model_path = BASE_DIR / "models" / "runs" / "detect" / "price_tag_detector-12" / "weights" / "best.pt"
video_path = BASE_DIR / "data" / "videos" / "25_12-20.mp4"

# Путь, куда сохраним результат
output_path = BASE_DIR / "data" / "videos" / "result_detection.mp4"

if not model_path.exists():
    print(f"Модель не найдена: {model_path}")
    exit()

model = YOLO(str(model_path))
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"Не удалось открыть видео: {video_path}")
else:
    # Получаем параметры исходного видео для записи
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Настраиваем объект для записи видео
    # Используем кодек 'mp4v' для формата .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    cv2.namedWindow('Detection Test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection Test', 1280, 720)

    print(f"Обработка началась. Результат сохранится в: {output_path}")
    print("Нажмите 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция
        results = model(frame, conf=0.15, verbose=False, device=0)
        annotated = results[0].plot()

        # 1. ЗАПИСЫВАЕМ кадр в файл (в оригинальном разрешении)
        out.write(annotated)

        # 2. ОТОБРАЖАЕМ уменьшенную копию для экрана
        display_width = 1280
        scale = display_width / annotated.shape[1]
        display_height = int(annotated.shape[0] * scale)
        display = cv2.resize(annotated, (display_width, display_height))

        cv2.imshow('Detection Test', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождаем все ресурсы
    cap.release()
    out.release()  # Важно закрыть файл, чтобы видео сохранилось корректно
    cv2.destroyAllWindows()
    print(f"Готово! Видео сохранено в {output_path}")
