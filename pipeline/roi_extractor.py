import json
import cv2


class ROIExtractor:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

    def extract_rois(self, image, template_name):
        template = self.config['templates'].get(template_name)

        if not template:
            return {}

        h, w = image.shape[:2]

        result = {}

        for roi_name, coords in template['roi'].items():
            x1 = int(coords[0] * w)
            y1 = int(coords[1] * h)
            x2 = int(coords[2] * w)
            y2 = int(coords[3] * h)

            crop = image[y1:y2, x1:x2]

            result[roi_name] = crop

        return result