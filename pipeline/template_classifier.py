import cv2
import numpy as np
from pathlib import Path
import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image

class TemplateClassifier:
    def __init__(self, onnx_path="models/format_classifier.onnx",
                 class_names_path="models/format_class_names.txt"):
        self.use_onnx = Path(onnx_path).exists()
        if self.use_onnx:
            self.session = ort.InferenceSession(onnx_path)
            self.input_name = self.session.get_inputs()[0].name
            with open(class_names_path) as f:
                self.class_names = [line.strip() for line in f]
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.class_names = ["6x6", "A4_horiz", "A4_vert", "A5"]

    def predict(self, image):
        """Принимает BGR numpy array, возвращает имя класса"""
        if self.use_onnx:
            pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil).unsqueeze(0).numpy()
            outputs = self.session.run(None, {self.input_name: input_tensor})
            return self.class_names[np.argmax(outputs[0])]
        else:
            # Fallback по геометрии
            h, w = image.shape[:2]
            ratio = w / h
            if 0.8 <= ratio <= 1.2:
                return "6x6"
            elif ratio > 2.5:
                return "A4_horiz"
            elif ratio > 1.5:
                return "A5"
            else:
                return "A4_vert"