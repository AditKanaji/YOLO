# Python In-built packages
from pathlib import Path
import streamlit as st
import cv2
import math

# External packages
from io import BytesIO
import tempfile
import numpy as np
from PIL import Image

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="PPE Detection using YOLO",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("PPE Detection using YOLO")

# Sidebar header
st.sidebar.header("Video Config")

# Sidebar options for video source
source_radio = st.sidebar.radio("Select Video Source", ["Upload Video", "Webcam URL"])
confidence = st.sidebar.slider("Detection Confidence", min_value=0.1, max_value=1.0, value=0.5)

# Function to determine color based on class
def get_color(current_class):
    if current_class in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
        return (0, 0, 255)  # Red for warnings
    elif current_class in ['Hardhat', 'Safety Vest', 'Mask']:
        return (0, 255, 0)  # Green for safety compliance
    else:
        return (255, 0, 0)  # Blue for other objects

# Function to perform detection and annotate frames
def detect_and_annotate_ppe(frame, model):
    # Perform object detection on the current frame
    results = model(frame, stream=True)

    # Class names for YOLO model
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
                  'Safety Vest']

    # Process detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            if cls >= 0 and cls < len(classNames):
                currentClass = classNames[cls]
                if conf > confidence:
                    myColor = get_color(currentClass)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), myColor, 3)

                    # Draw label background
                    label = f'{currentClass} {conf:.2f}'
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), myColor, cv2.FILLED)

                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

# Function to handle video streaming and processing
def process_video(video_source):
    cap = cv2.VideoCapture(video_source)

    # Load the YOLO model
    model = helper.load_model("weights/ppe.pt")

    if not cap.isOpened():
        st.error("Error opening video stream")
        return

    # Create a placeholder for the video frame
    stframe = st.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break  # End of video or stream

        # Perform detection and annotate the frame
        annotated_frame = detect_and_annotate_ppe(frame, model)

        # Convert the image from OpenCV format to PIL format
        img_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        # Display the output frame with detections using Streamlit
        stframe.image(img_pil, caption='Processed Frame', use_column_width=True)

    # Release video capture and close windows
    cap.release()

# Main logic for handling video source and detection
if source_radio == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi"])

    if uploaded_file is not None:
        # Use a temporary directory to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Display the uploaded video
        st.video(temp_file_path)

        if st.sidebar.button('Start PPE Detection'):
            process_video(temp_file_path)
    else:
        st.warning("Please upload a video file.")

elif source_radio == "Webcam URL":
    webcam_url = st.sidebar.text_input("Enter Webcam URL")

    if webcam_url:
        if st.sidebar.button('Start PPE Detection'):
            process_video(webcam_url)
    else:
        st.warning("Please enter a valid webcam URL.")
