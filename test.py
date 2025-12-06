import cv2
from ultralytics import YOLO
import os

def main():
    model_path = r"C:\Users\raara\OneDrive\Desktop\Ashwita-ScienceFair\new\yolo11n.pt"
    model = YOLO(model_path)

    test_dir = r"C:\Users\raara\OneDrive\Desktop\Ashwita-ScienceFair\new\test"
    output_dir = r"C:\Users\raara\OneDrive\Desktop\Ashwita-ScienceFair\new\test_results"
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)

        results = model(img_path)

        # Save results
        for r in results:
            output_img = r.plot()
            cv2.imwrite(os.path.join(output_dir, img_name), output_img)

        print(f"Inference done for: {img_name}")

if __name__ == "__main__":
    main()
