import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

from pipeline.aligner import PriceTagAligner
from pipeline.template_classifier import TemplateClassifier
from pipeline.roi_extractor import ROIExtractor
from pipeline.ocr_reader import OCRReader
from pipeline.qr_reader import QRReader
from pipeline.validator import Validator
from pipeline.temporal_fusion import TemporalFusion
from pipeline.iou_tracker import IoUTracker

def process_video(video_path, output_dir="results",
                  detector_path="models/runs/detect/price_tag_detector/weights/best.pt",
                  roi_config="config/roi_templates.json"):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 2)  # каждые 2 секунды
    detector = YOLO(detector_path)
    aligner = PriceTagAligner()
    classifier = TemplateClassifier()
    extractor = ROIExtractor(roi_config)
    ocr = OCRReader()
    qr_reader = QRReader()
    validator = Validator()
    fusion = TemporalFusion()
    tracker = IoUTracker()

    results = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            detections = detector(frame, conf=0.15, verbose=False)
            boxes = []
            for box in detections[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))

            # Трекинг
            tracks = tracker.update(boxes)

            for track_id, bbox in tracks.items():
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # Выравнивание
                aligned = aligner.align(crop)
                # Классификация формата
                template = classifier.predict(aligned)
                # Извлечение ROI
                rois = extractor.extract_rois(aligned, template)
                qr_zone = extractor.get_qr_zone(aligned, template)

                # OCR
                data = {}
                data['product_name'] = ocr.read(rois.get('product_name'))
                data['price_default'] = ocr.read_price(rois.get('price_default'))
                data['price_card'] = ocr.read_price(rois.get('price_card'))
                data['price_discount'] = ocr.read_price(rois.get('price_discount'))
                data['id_sku'] = ocr.read_id_sku(rois.get('id_sku'))
                data['barcode'] = ocr.read_barcode(rois.get('barcode'))
                data['discount_amount'] = ocr.read(rois.get('discount_amount'))
                data['print_datetime'] = ocr.read(rois.get('print_datetime'))
                data['code'] = ocr.read(rois.get('code'))
                data['additional_info'] = ocr.read(rois.get('additional_info'))
                data['special_symbols'] = ocr.read(rois.get('special_symbols'))

                # QR
                qr_data = qr_reader.read(qr_zone)
                data.update(qr_data)  # добавляет поля, например qr_code_barcode

                # Валидация
                data = validator.validate(data)

                # Добавляем координаты и время
                timestamp_ms = (frame_idx / fps) * 1000
                data['filename'] = video_path.name
                data['frame_timestamp'] = timestamp_ms
                data['x_min'], data['y_min'], data['x_max'], data['y_max'] = x1, y1, x2, y2

                # Сохраняем наблюдение в temporal fusion
                fusion.add(track_id, data)

            # Удаляем потерянные треки и записываем результат
            lost_tracks = [tid for tid, t in tracker.tracks.items() if t['lost'] > tracker.max_lost]
            for tid in lost_tracks:
                if tid in fusion.tracks:
                    fused_data = fusion.fuse(tid)
                    if fused_data:
                        results.append(fused_data)
                    del fusion.tracks[tid]

        frame_idx += 1

    cap.release()
    # Сохраняем CSV
    if results:
        df = pd.DataFrame(results)
        output_path = Path(output_dir) / f"{video_path.stem}_results.csv"
        df.to_csv(output_path, index=False, quoting=1)  # QUOTE_ALL
        return output_path
    return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <video_path>")
    else:
        process_video(Path(sys.argv[1]))