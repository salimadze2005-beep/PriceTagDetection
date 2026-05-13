import cv2
import json
from pathlib import Path
from ultralytics import YOLO
from pipeline.aligner import PriceTagAligner
from pipeline.template_classifier import TemplateClassifier
from pipeline.roi_extractor import ROIExtractor

def check_roi(video_path, roi_config, detector_path, frame_number=100):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Не удалось прочитать кадр")
        return

    detector = YOLO(detector_path)
    aligner = PriceTagAligner()
    classifier = TemplateClassifier()
    extractor = ROIExtractor(roi_config)

    results = detector(frame, conf=0.80, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("Ценники не найдены")
        return

    # Выбираем самый крупный бокс
    max_area = 0
    best_box = None
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        return

    x1, y1, x2, y2 = best_box
    crop = frame[y1:y2, x1:x2]
    cv2.imwrite("test_crop_original.jpg", crop)

    aligned = aligner.align(crop)
    cv2.imwrite("test_crop_aligned.jpg", aligned)

    template = classifier.predict(aligned)
    print(f"Определённый формат: {template}")

    rois = extractor.extract_rois(aligned, template)

    for field, roi_img in rois.items():
        if roi_img is not None and roi_img.size > 0:
            filename = f"test_roi_{field}.jpg"
            cv2.imwrite(filename, roi_img)
            print(f"Сохранена зона '{field}' -> {filename}")
        else:
            print(f"Зона '{field}' пустая или не определена")

    qr_zone = extractor.get_qr_zone(aligned, template)
    if qr_zone is not None:
        cv2.imwrite("test_roi_qr_zone.jpg", qr_zone)

if __name__ == "__main__":
    # Пути
    project_root = Path(__file__).resolve().parent.parent
    video = project_root / "data" / "videos" / "25_12-20.mp4"  # замени на своё
    roi_config = project_root / "config" / "roi_templates.json"
    detector = project_root / "models" / "runs" / "detect" / "price_tag_detector" / "weights" / "best.pt"  # подставь актуальный путь

    check_roi(str(video), str(roi_config), str(detector), frame_number=100)