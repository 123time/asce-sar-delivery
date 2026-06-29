"""Training entry point for the delivered YOLO11 model."""

from ultralytics import YOLO


MODEL_CFG = "ultralytics/cfg/models/11/yolo11-ADown-Backbone-SOEP1EMA-SWC-DySample-fixed.yaml"
DATA_CFG = "path/to/data.yaml"


def main():
    model = YOLO(MODEL_CFG)
    model.train(
        data=DATA_CFG,
        imgsz=640,
        epochs=300,
        batch=32,
        workers=4,
        optimizer="SGD",
        project="runs/train",
        name="exp",
        amp=False,
    )


if __name__ == "__main__":
    main()
