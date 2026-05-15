import cv2
import numpy as np
import pandas as pd


class BestFrameSelector:
    def __init__(self):
        self.tracks = {}

    def sharpness_score(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

    def bbox_score(self, bbox):
        x1, y1, x2, y2 = bbox

        return (x2 - x1) * (y2 - y1)

    def total_score(self, crop, bbox):
        sharpness = self.sharpness_score(crop)

        area = self.bbox_score(bbox)

        return sharpness * 0.7 + area * 0.3

    def update(
        self,
        track_id,
        frame,
        bbox,
        timestamp,
        data
    ):
        x1, y1, x2, y2 = map(int, bbox)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return

        score = self.total_score(crop, bbox)

        if track_id not in self.tracks:
            self.tracks[track_id] = {
                'score': score,
                'timestamp': timestamp,
                'bbox': bbox,
                'data': data
            }

        elif score > self.tracks[track_id]['score']:
            self.tracks[track_id] = {
                'score': score,
                'timestamp': timestamp,
                'bbox': bbox,
                'data': data
            }

    def export_csv(self, output_path):
        rows = []

        for track_id, item in self.tracks.items():
            row = item['data']

            row['frame_timestamp'] = item['timestamp']

            x1, y1, x2, y2 = item['bbox']

            row['x_min'] = x1
            row['y_min'] = y1
            row['x_max'] = x2
            row['y_max'] = y2

            rows.append(row)

        df = pd.DataFrame(rows)

        df.to_csv(
            output_path,
            index=False,
            encoding='utf-8'
        )