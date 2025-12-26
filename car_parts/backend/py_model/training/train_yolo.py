from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="/content/drive/MyDrive/dataset/data.yaml",  # Colab 기준
        epochs=5,
        imgsz=640,
        batch=16,
        project="/content/drive/MyDrive/runs",
        name="train_exp1"
    )

if __name__ == "__main__":
    main()
