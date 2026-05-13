from pyzbar.pyzbar import decode
import cv2
import re

class QRReader:
    def read(self, image):
        """Возвращает словарь данных QR. Поддерживает разные форматы."""
        if image is None: return {}
        # Прямое чтение
        decoded = decode(image)
        if not decoded:
            # Попытка с бинаризацией
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            decoded = decode(thresh)
        if not decoded:
            return {}

        data = decoded[0].data.decode('utf-8').strip()
        return self._parse(data)

    def _parse(self, qr_str):
        """Парсит строку QR в словарь."""
        result = {}
        # Попытка & или ; как разделители
        pairs = re.split(r'[&;\n]', qr_str)
        for pair in pairs:
            if '=' in pair:
                k, v = pair.split('=', 1)
                result[k.strip()] = v.strip()
            elif pair.strip().isdigit() and len(pair.strip()) == 13:
                result['barcode'] = pair.strip()
        # Если ничего не нашли, сохраняем как сырую строку
        if not result:
            result['raw'] = qr_str
        return result