"""
Download YOLOv8 model for object detection
Run this script to download a pre-trained YOLOv8 model
"""
from ultralytics import YOLO
import os

# Download YOLOv8 nano model (smallest, fastest)
print("Downloading YOLOv8 model...")
model = YOLO('yolov8n.pt')  # This will auto-download

# Save to model folder
model_dir = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(model_dir, exist_ok=True)

output_path = os.path.join(model_dir, 'best.pt')
print(f"Saving model to: {output_path}")

# Copy the downloaded model
import shutil
yolo_cache = os.path.join(os.path.expanduser('~'), '.cache', 'ultralytics', 'yolov8n.pt')
if os.path.exists(yolo_cache):
    shutil.copy(yolo_cache, output_path)
    print(f"✓ Model saved successfully to {output_path}")
else:
    print(f"Model downloaded to cache. Manually copy from: {yolo_cache}")
    print(f"To: {output_path}")
