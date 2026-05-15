from pyzbar.pyzbar import decode
import cv2
import numpy as np


class QRReader:
    def __init__(self):
        pass

    def preprocess_versions(self, image):
        versions = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        versions.append(gray)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        versions.append(enhanced)

        thresh = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5
        )

        versions.append(thresh)

        sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        sharp = cv2.filter2D(
            enhanced,
            -1,
            sharpen_kernel
        )

        versions.append(sharp)

        return versions

    def read(self, image):
        versions = self.preprocess_versions(image)

        for img in versions:
            decoded = decode(img)

            if decoded:
                try:
                    return decoded[0].data.decode('utf-8')
                except:
                    pass

        return None