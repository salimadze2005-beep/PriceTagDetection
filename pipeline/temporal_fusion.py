from collections import Counter

class TemporalFusion:
    def __init__(self):
        self.tracks = {}

    def add(self, track_id, data):
        if track_id not in self.tracks:
            self.tracks[track_id] = []
        self.tracks[track_id].append(data)

    def fuse(self, track_id):
        """Возвращает объединённый словарь с majority vote для каждого поля."""
        if track_id not in self.tracks:
            return {}
        all_observations = self.tracks[track_id]
        # Собираем все ключи
        all_keys = set()
        for obs in all_observations:
            all_keys.update(obs.keys())

        result = {}
        for key in all_keys:
            values = [obs.get(key) for obs in all_observations if obs.get(key) is not None]
            if values:
                # Голосование
                result[key] = Counter(values).most_common(1)[0][0]
        return result