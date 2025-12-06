from ultralytics import YOLO

# Load YOLOv8 base model
model = YOLO("yolov8n.pt")  # nano version (fast & light)

# Train using your dataset
model.train(
    data="C:/Users/raara/OneDrive/Desktop/Ashwita-ScienceFair/Ashwita_Project/TrashNet-main/data.yaml",
    epochs=1,
    imgsz=640,
    batch=16
)

# Save final trained model
model.export()   # ensures .pt format
