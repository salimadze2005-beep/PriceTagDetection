import json
from pathlib import Path

class ROIExtractor:
    def __init__(self, config_path="config/roi_templates.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

    def extract_rois(self, image, template_name):
        """Возвращает словарь ROI-изображений для заданного шаблона."""
        h, w = image.shape[:2]
        rois = {}
        template = self.config['templates'].get(template_name)
        if not template:
            return rois
        for field, roi in template['roi'].items():
            if roi is None: continue
            x1 = int(roi[0] * w)
            y1 = int(roi[1] * h)
            x2 = int(roi[2] * w)
            y2 = int(roi[3] * h)
            if x2 > x1 and y2 > y1:
                rois[field] = image[y1:y2, x1:x2]
        return rois

    def get_qr_zone(self, image, template_name):
        """Возвращает изображение QR-зоны или None."""
        template = self.config['templates'].get(template_name)
        if not template or 'qr_zone' not in template or template['qr_zone'] is None:
            return None
        h, w = image.shape[:2]
        qr = template['qr_zone']
        x1 = int(qr[0] * w)
        y1 = int(qr[1] * h)
        x2 = int(qr[2] * w)
        y2 = int(qr[3] * h)
        if x2 > x1 and y2 > y1:
            return image[y1:y2, x1:x2]
        return None