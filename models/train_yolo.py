from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    data_yaml = project_root / "real_dataset" / "dataset.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Датасет не найден: {data_yaml}")

    model = YOLO('yolov8s.pt')
    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        workers=0,          # Windows fix
        cache=True,
        degrees=0,          # без поворота
        scale=0.1,
        perspective=0.0005,
        mosaic=0.2,
        mixup=0.0,
        copy_paste=0.0,
        patience=10,
        name='price_tag_detector'
    )
    model.export(format='onnx')