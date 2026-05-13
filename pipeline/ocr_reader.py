import pytesseract
import re
import cv2
import numpy as np

# Путь к tesseract.exe – подкорректируй, если ставил в другое место
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class OCRReader:
    def __init__(self):
        self.config = '--psm 6 -l rus+eng'  # режим: uniform block of text, русский+английский

    def read(self, image):
        """Возвращает весь текст с изображения одной строкой."""
        if image is None or image.size == 0:
            return ""
        # Предобработка: бинаризация для улучшения OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config=self.config)
        return ' '.join(text.split())

    def read_id_sku(self, image):
        text = self.read(image)
        matches = re.findall(r'\b\d{9,15}\b', text)
        return matches[0] if matches else None

    def read_barcode(self, image):
        text = self.read(image)
        matches = re.findall(r'\b\d{12,14}\b', text)
        return matches[0] if matches else None

    def read_price(self, image):
        text = self.read(image)
        m = re.search(r'(\d+)[,.](\d{2})', text)
        return float(f"{m.group(1)}.{m.group(2)}") if m else None