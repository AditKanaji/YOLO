# Python In-built packages
from pathlib import Path
import PIL
import tempfile  # Import tempfile
import time  # Import time for frame processing

# External packages
import streamlit as st
import cv2
import numpy as np

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")

# Sidebar header
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'PPE Detection', 'License Plate Detection with EasyOCR', 'Car Counting'])

# Sidebar slider
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Additional sliders for License Plate Detection with EasyOCR
if model_type == 'License Plate Detection with EasyOCR':
    floodfill_threshold = st.sidebar.slider('Floodfill Threshold', 0, 250, 100, step=1)
    threshold_block_size = st.sidebar.slider('Threshold Block Size (odd number, > 1)', 3, 201, 101, step=2)
    brightness = st.sidebar.slider('Brightness Adjustment', -100, 100, 0, step=1)

# Cache the model loading function
@st.cache_resource
def load_model(model_type):
    model_path = get_model_path(model_type)
    model = helper.load_model(model_path)
    return model

def get_model_path(model_type):
    if model_type == 'Detection':
        return Path(settings.DETECTION_MODEL)
    elif model_type == 'PPE Detection':
        return Path(settings.CUSTOM_MODEL3)
    elif model_type == 'License Plate Detection with EasyOCR':
        return Path(settings.CUSTOM_MODEL1)
    elif model_type == 'Car Counting':
        return Path(settings.CUSTOM_MODEL1)  # Make sure this line is correct and the model path is valid
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# Load ML model based on selected task
model = load_model(model_type)
# st.write("Model loaded:", model)

# Sidebar: Image/Video Config
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST + ['Webcam', 'CCTV URL', 'YouTube'])

def calculate_centroid(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)

def is_new_track(centroid, existing_tracks, threshold=50):
    for track_id, track_centroid in existing_tracks.items():
        distance = np.sqrt((centroid[0] - track_centroid[0])**2 + (centroid[1] - track_centroid[1])**2)
        if distance < threshold:
            return False
    return True


def count_cars_and_draw_boxes(frame, model, existing_tracks, confidence_threshold=0.5):
    height, width, _ = frame.shape
    bottom_half_y = height // 2  # Y-coordinate that represents the bottom half of the frame

    results = model(frame, stream=True)
    detected_count = 0
    vehicle_class_ids = [0]  # Assuming class_id 0 is for cars

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            centroid = calculate_centroid([x1, y1, x2, y2])

            # Check if the bounding box is in the bottom half of the frame
            if y2 > bottom_half_y and cls in vehicle_class_ids and conf > confidence_threshold:
                if is_new_track(centroid, existing_tracks):
                    detected_count += 1
                    new_track_id = len(existing_tracks) + 1
                    existing_tracks[new_track_id] = centroid
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for new cars
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for counted cars
    return detected_count


# Process image if selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        if source_img is None:
            default_image_path = str(settings.DEFAULT_IMAGE)
            default_image = PIL.Image.open(default_image_path)
            st.image(default_image_path, caption="Default Image", use_column_width=True)
        else:
            uploaded_image = PIL.Image.open(source_img)
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image', use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                if model_type == 'License Plate Detection with EasyOCR':
                    original_image = np.array(uploaded_image)
                    license_plates = model.predict(original_image)

                    if license_plates and license_plates[0].boxes:
                        for i, license_plate in enumerate(license_plates[0].boxes):
                            x1, y1, x2, y2 = license_plate.xyxy[0]
                            license_plate_image = original_image[int(y1):int(y2), int(x1):int(x2)]
                            processed_license_plate = helper.process_license_plate(license_plate_image,
                                                                                   floodfill_threshold,
                                                                                   threshold_block_size, brightness)

                            detections = reader.readtext(processed_license_plate)
                            if detections:
                                detected_plate_text = detections[0][1]
                                cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                text_size, _ = cv2.getTextSize(detected_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 2)
                                text_x = int(x1 + (x2 - x1 - text_size[0]) / 2)
                                text_y = int(y1 - 10)
                                background_width = text_size[0] + 20
                                background_height = text_size[1] + 10
                                background_x1 = text_x - 10
                                background_y1 = text_y - text_size[1] - 5
                                background_x2 = background_x1 + background_width
                                background_y2 = text_y + 5
                                background_color = (0, 0, 0)
                                cv2.rectangle(original_image, (background_x1, background_y1),
                                              (background_x2, background_y2), background_color, -1)
                                text_color = (255, 255, 255)
                                cv2.putText(original_image, detected_plate_text, (text_x, text_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, text_color, 2)

                        st.image(original_image, caption='Detected License Plate', use_column_width=True)
                        st.image(processed_license_plate, caption='Processed License Plate', use_column_width=True)
                    else:
                        st.error("No license plates detected in the image.")

                else:
                    res = model.predict(np.array(uploaded_image), conf=confidence)
                    boxes = res[0].boxes

                    if len(boxes) == 0:
                        st.error("No objects detected in the image.")
                    else:
                        res_plotted = res[0].plot()[:, :, ::-1]
                        st.image(res_plotted, caption='Detected Image', use_column_width=True)

                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)

elif source_radio == settings.VIDEO:
    if model_type == 'Car Counting':
        uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            if st.sidebar.button('Start Detection'):
                cap = cv2.VideoCapture(video_path)
                existing_tracks = {}
                total_car_count = 0

                # Create a placeholder for the video frame
                stframe = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Car Counting
                    car_count = count_cars_and_draw_boxes(frame, model, existing_tracks, confidence)
                    total_car_count += car_count

                    # Display the frame with bounding boxes
                    stframe.image(frame, channels="BGR")

                    # To make sure the video displays in real-time
                    time.sleep(0.03)  # Adjust the sleep time as necessary

                cap.release()
                st.write(f"Total Cars Counted: {total_car_count}")
    else:
        uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            if st.sidebar.button('Start Detection'):
                cap = cv2.VideoCapture(video_path)

                # Create a placeholder for the video frame
                stframe = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # YOLO object detection
                    res = model(frame)
                    res_plotted = res[0].plot()

                    # Display the frame with bounding boxes
                    stframe.image(res_plotted, channels="BGR")

                    # To make sure the video displays in real-time
                    time.sleep(0.03)  # Adjust the sleep time as necessary

                cap.release()

elif source_radio == 'Webcam':
    if st.sidebar.button('Start Webcam Detection'):
        cap = cv2.VideoCapture(0)  # Open webcam

        # Create a placeholder for the video frame
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break

            # YOLO object detection
            res = model(frame)
            res_plotted = res[0].plot()

            # Display the frame with bounding boxes
            stframe.image(res_plotted, channels="BGR")

            # To make sure the video displays in real-time
            time.sleep(0.03)  # Adjust the sleep time as necessary

        cap.release()

elif source_radio == 'CCTV URL':
    url = st.sidebar.text_input("Enter CCTV URL:")
    if url and st.sidebar.button('Start CCTV Detection'):
        cap = cv2.VideoCapture(url)

        # Create a placeholder for the video frame
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from CCTV.")
                break

            # YOLO object detection
            res = model(frame)
            res_plotted = res[0].plot()

            # Display the frame with bounding boxes
            stframe.image(res_plotted, channels="BGR")

            # To make sure the video displays in real-time
            time.sleep(0.03)  # Adjust the sleep time as necessary

        cap.release()

elif source_radio == 'YouTube':
    video_url = st.sidebar.text_input("Enter YouTube video URL:")
    if video_url and st.sidebar.button('Start YouTube Detection'):
        cap = cv2.VideoCapture(video_url)

        # Create a placeholder for the video frame
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from YouTube.")
                break

            # YOLO object detection
            res = model(frame)
            res_plotted = res[0].plot()

            # Display the frame with bounding boxes
            stframe.image(res_plotted, channels="BGR")

            # To make sure the video displays in real-time
            time.sleep(0.03)  # Adjust the sleep time as necessary

        cap.release()

# About
st.sidebar.markdown(settings.ABOUT)

