import easyocr
import cv2
import re

class OCRReader:

    def __init__(self):

        self.reader = easyocr.Reader(
            ['en', 'ru'],
            gpu=True
        )

    def preprocess(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2
        )

        gray = cv2.GaussianBlur(
            gray,
            (3,3),
            0
        )

        _, thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return thresh

    def read_text(self, image):

        prep = self.preprocess(image)

        result = self.reader.readtext(
            prep,
            detail=0,
            paragraph=False
        )

        return " ".join(result)

    def read_digits(self, image):

        text = self.read_text(image)

        digits = re.findall(r'\d+', text)

        return ''.join(digits)