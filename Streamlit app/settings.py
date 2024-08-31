from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file (main file: /yolov8-streamlit)
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the main folder
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'

SOURCES_LIST = [IMAGE, VIDEO]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / '1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / '2.jpg'

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'
CUSTOM_MODEL1 =  'ocr/license_plate_detector.pt'
CUSTOM_MODEL2 = MODEL_DIR / 'car.pt'
CUSTOM_MODEL3 = MODEL_DIR / 'ppe.pt'

# About information
ABOUT = """
### Object Detection App
This app demonstrates the use of the YOLOv8 model for object detection tasks. 
You can upload images or videos, or use a webcam, CCTV, or YouTube link as a source.

- **Detection:** Detect various objects using the standard YOLOv8 model.
- **PPE Detection:** Detect personal protective equipment (PPE) on workers.
- **License Plate Detection:** Detect and read license plates using EasyOCR.
- **Car Counting:** Count the number of cars in a video feed.

**Source Code:** [GitHub Link](https://github.com/your-repo)
"""
