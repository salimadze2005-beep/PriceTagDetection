from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="field_synth/dataset.yaml",

    epochs=50,

    imgsz=640,

    batch=16,

    workers=0,

    cache=True,

    degrees=0,

    scale=0.08,

    perspective=0.0005,

    mosaic=0.15,

    mixup=0.0,

    copy_paste=0.0,

    patience=10,

    name="field_detector"
)

model.export(format="onnx")