import cv2
import numpy as np

class PriceTagAligner:
    def align(self, crop):
        """
        Выравнивает перспективу ценника по его контуру.
        Возвращает выровненное изображение (горизонтально ориентированное).
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return crop

        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # intp вместо int32 для совместимости

        width = int(rect[1][0])
        height = int(rect[1][1])

        if width < 10 or height < 10:
            return crop

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        src = np.array(box, dtype="float32")
        matrix = cv2.getPerspectiveTransform(src, dst)
        aligned = cv2.warpPerspective(crop, matrix, (width, height))

        return aligned