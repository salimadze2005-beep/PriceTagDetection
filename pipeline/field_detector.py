

from ultralytics import YOLO

CLASSES = [
    "product_name",
    "price",
    "barcode",
    "id_sku"
]

class FieldDetector:

    def __init__(self, model_path):

        self.model = YOLO(model_path)

    def detect(self, image):

        results = self.model(
            image,
            conf=0.20,
            verbose=False
        )

        rois = {}

        for box in results[0].boxes:

            cls_id = int(box.cls[0])

            cls_name = CLASSES[cls_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            roi = image[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # оставляем biggest bbox
            if cls_name in rois:

                old_area = (
                    rois[cls_name].shape[0] *
                    rois[cls_name].shape[1]
                )

                new_area = (
                    roi.shape[0] *
                    roi.shape[1]
                )

                if new_area > old_area:
                    rois[cls_name] = roi

            else:
                rois[cls_name] = roi

        return rois