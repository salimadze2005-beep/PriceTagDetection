# run_pipeline.py

import cv2
import pandas as pd

from pathlib import Path
from ultralytics import YOLO

from pipeline.aligner import PriceTagAligner
from pipeline.template_classifier import TemplateClassifier
from pipeline.roi_extractor import ROIExtractor
from pipeline.ocr_reader import OCRReader
from pipeline.qr_reader import QRReader
from pipeline.validator import Validator
from pipeline.best_frame_selector import BestFrameSelector


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / 'models'
    / 'runs'
    / 'detect'
    / 'price_tag_detector'
    / 'weights'
    / 'best.pt'
)

VIDEO_PATH = (
    BASE_DIR
    / 'data'
    / 'videos'
    / '26_12-20.mp4'
)

ROI_CONFIG = (
    BASE_DIR
    / 'config'
    / 'roi_templates.json'
)

OUTPUT_CSV = (
    BASE_DIR
    / 'results.csv'
)


def preprocess_barcode_roi(image):

    versions = []

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    versions.append(gray)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    versions.append(enhanced)

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )

    versions.append(adaptive)

    sharpen_kernel = [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]

    sharpened = cv2.filter2D(
        enhanced,
        -1,
        sharpen_kernel
    )

    versions.append(sharpened)

    return versions


def main():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f'Не найдена модель: {MODEL_PATH}'
        )

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f'Не найдено видео: {VIDEO_PATH}'
        )

    print('Загрузка модели...')

    model = YOLO(str(MODEL_PATH))

    print('Инициализация модулей...')

    aligner = PriceTagAligner()

    classifier = TemplateClassifier()

    extractor = ROIExtractor(
        ROI_CONFIG
    )

    ocr = OCRReader()

    qr_reader = QRReader()

    validator = Validator()

    selector = BestFrameSelector()

    print('Открытие видео...')

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            'Не удалось открыть видео'
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(
        f'FPS: {fps}'
    )

    print(
        f'Frames: {total_frames}'
    )

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        print(
            f'Frame {frame_id}/{total_frames}',
            end='\r'
        )

        results = model.track(
            frame,

            conf=0.15,

            persist=True,

            tracker='bytetrack.yaml',

            verbose=False
        )

        boxes = results[0].boxes

        if boxes.id is None:

            frame_id += 1

            continue

        for box, track_id in zip(
            boxes,
            boxes.id
        ):

            track_id = int(
                track_id.item()
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                frame.shape[1],
                x2
            )

            y2 = min(
                frame.shape[0],
                y2
            )

            crop = frame[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            aligned = aligner.align(
                crop
            )

            if aligned is None:
                continue

            if aligned.size == 0:
                continue

            template_name = (
                classifier.predict(
                    aligned
                )
            )

            rois = extractor.extract_rois(
                aligned,
                template_name
            )

            result = {

                'filename': VIDEO_PATH.name,

                'barcode': None,

                'qr_code_barcode': None,

                'id_sku': None,

                'price_default': None,

                'price_card': None
            }

            # BARCODE

            if 'barcode' in rois:

                barcode_roi = rois['barcode']

                barcode_value = None

                processed_versions = (
                    preprocess_barcode_roi(
                        barcode_roi
                    )
                )

                for processed in processed_versions:

                    temp_roi = cv2.cvtColor(
                        processed,
                        cv2.COLOR_GRAY2BGR
                    )

                    barcode_value = (
                        ocr.read_barcode(
                            temp_roi
                        )
                    )

                    if barcode_value:
                        break

                result['barcode'] = (
                    barcode_value
                )

            # ID SKU

            if 'id_sku' in rois:

                result['id_sku'] = (
                    ocr.read_id_sku(
                        rois['id_sku']
                    )
                )

            # PRICE DEFAULT

            if 'price_default' in rois:

                result['price_default'] = (
                    ocr.read(
                        rois['price_default']
                    )
                )

            # PRICE CARD

            if 'price_card' in rois:

                result['price_card'] = (
                    ocr.read(
                        rois['price_card']
                    )
                )

            # QR

            if 'qr' in rois:

                qr_result = qr_reader.read(
                    rois['qr']
                )

                if (
                    'barcode'
                    in qr_result
                ):

                    result[
                        'qr_code_barcode'
                    ] = (
                        qr_result['barcode']
                    )

            # VALIDATION

            result = validator.validate(
                result
            )

            # TIMESTAMP

            timestamp = int(
                (frame_id / fps) * 1000
            )

            # BEST FRAME SELECTION

            selector.update(

                track_id=track_id,

                image=aligned,

                result=result,

                timestamp=timestamp,

                bbox=(
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

        frame_id += 1

    cap.release()

    print('\nЭкспорт результатов...')

    final_results = (
        selector.export_results()
    )

    df = pd.DataFrame(
        final_results
    )

    df.to_csv(

        OUTPUT_CSV,

        index=False,

        encoding='utf-8'
    )

    print(
        f'CSV сохранён: {OUTPUT_CSV}'
    )

    print(
        f'Всего строк: {len(df)}'
    )


if __name__ == '__main__':

    main()