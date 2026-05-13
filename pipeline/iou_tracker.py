import numpy as np

class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_lost=5):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks = {}  # track_id -> {'bbox': (x1,y1,x2,y2), 'lost': 0}
        self.next_id = 0

    def update(self, detections):
        """
        detections: список bounding box'ов [(x1,y1,x2,y2), ...]
        Возвращает словарь {track_id: (x1,y1,x2,y2)} для активных треков.
        """
        assigned = {}
        unassigned_det = list(enumerate(detections))
        track_ids = list(self.tracks.keys())

        # Сброс флагов lost
        for tid in track_ids:
            self.tracks[tid]['lost'] += 1

        if not track_ids or not unassigned_det:
            # Все новые или нет треков
            for i, bbox in unassigned_det:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {'bbox': bbox, 'lost': 0}
                assigned[new_id] = bbox
            # Удаление потерянных
            self._cleanup()
            return assigned

        # Матрица IoU
        ious = np.zeros((len(unassigned_det), len(track_ids)))
        for i, (di, det_bbox) in enumerate(unassigned_det):
            for j, tid in enumerate(track_ids):
                ious[i, j] = self._iou(det_bbox, self.tracks[tid]['bbox'])

        # Жадное сопоставление
        while len(unassigned_det) > 0 and len(track_ids) > 0:
            i, j = np.unravel_index(ious.argmax(), ious.shape)
            if ious[i, j] < self.iou_threshold:
                break
            det_idx = unassigned_det[i][0]
            tid = track_ids[j]
            self.tracks[tid]['bbox'] = detections[det_idx]
            self.tracks[tid]['lost'] = 0
            assigned[tid] = detections[det_idx]
            # Удаляем из матрицы
            unassigned_det.pop(i)
            track_ids.pop(j)
            ious = np.delete(ious, i, axis=0)
            ious = np.delete(ious, j, axis=1)

        # Создаём новые треки для оставшихся детекций
        for i, _ in unassigned_det:
            new_id = self.next_id
            self.next_id += 1
            bbox = detections[i]
            self.tracks[new_id] = {'bbox': bbox, 'lost': 0}
            assigned[new_id] = bbox

        self._cleanup()
        return assigned

    def _iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def _cleanup(self):
        to_remove = [tid for tid, data in self.tracks.items() if data['lost'] > self.max_lost]
        for tid in to_remove:
            del self.tracks[tid]