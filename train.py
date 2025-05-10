from ultralytics import YOLO
import argparse


def train_model(model_path: str = "yolov8m.pt", data_path: str = "./training_dataset/data.yaml", epochs: int = 100, batch_size: int = 2):
    model = YOLO(model_path)
    model.train(data=data_path, batch=batch_size, epochs=epochs, imgsz=640, project="training_result", fliplr=0.0, flipud=0.0, translate=0.0, scale=0.0, mosaic=0.0, single_cls=True)

def validate_model(model_path: str = "training_result/train/weights/best.pt", data_path: str = "./training_dataset/data.yaml"):
    model = YOLO(model_path)
    validation_results = model.val(data=data_path, imgsz=640, batch=1, conf=0.7, iou=0.7, device="0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate YOLOv8 model")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Path to the model")
    parser.add_argument("--data", type=str, default="./training_dataset/data.yaml", help="Path to the data yaml file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--validate", action="store_true", help="Run validation after training")
    parser.add_argument("--val-model", type=str, default=None, help="Path to model for validation (defaults to best training weights)")
    
    args = parser.parse_args()
    
    train_model(
        model_path=args.model,
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if args.validate:
        val_model = args.val_model if args.val_model else "training_result/train/weights/best.pt"
        validate_model(model_path=val_model, data_path=args.data)

