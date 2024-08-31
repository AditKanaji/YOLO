import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import tempfile
import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def license_complies_format(text):
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False

def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None

def process_frame(frame, yolo_model, reader):
    results = yolo_model(frame)
    detected = results[0]
    boxes = detected.boxes.xyxy.cpu().numpy()
    scores = detected.boxes.conf.cpu().numpy()
    class_ids = detected.boxes.cls.cpu().numpy()

    license_plate_results = []
    for i in range(len(boxes)):
        if int(class_ids[i]) == 0:  # Assuming class_id 0 is for license plates
            x1, y1, x2, y2 = map(int, boxes[i])
            crop = frame[y1:y2, x1:x2]
            text, score = read_license_plate(crop)
            if text:
                license_plate_results.append((x1, y1, x2, y2, text, score))

    return license_plate_results

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img

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

def count_cars_and_draw_boxes(frame, yolo_model, existing_tracks):
    results = yolo_model(frame)
    detected = results[0]
    boxes = detected.boxes.xyxy.cpu().numpy()
    class_ids = detected.boxes.cls.cpu().numpy()

    car_count = 0
    vehicle_class_ids = [0]  # Assuming class_id 0 is for cars, adjust if necessary

    for i in range(len(boxes)):
        if int(class_ids[i]) in vehicle_class_ids:
            centroid = calculate_centroid(boxes[i])
            x1, y1, x2, y2 = map(int, boxes[i])

            if is_new_track(centroid, existing_tracks):
                car_count += 1
                new_track_id = len(existing_tracks) + 1
                existing_tracks[new_track_id] = centroid
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for new cars
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for counted cars

    return car_count

def main():
    st.title("License Plate Detection and Car Counting")
    st.write("Upload a video file for license plate detection and car counting")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        video = cv2.VideoCapture(video_path)
        yolo_model = YOLO('ocr/license_plate_detector.pt')

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))

        # Create a temporary file to save the processed video
        processed_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (int(video.get(3)), int(video.get(4))))

        total_car_count = 0
        existing_tracks = {}

        for frame_no in range(frame_count):
            ret, frame = video.read()
            if not ret:
                break

            license_plate_results = process_frame(frame, yolo_model, reader)

            for result in license_plate_results:
                x1, y1, x2, y2, text, score = result
                draw_border(frame, (x1, y1), (x2, y2))
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            car_count = count_cars_and_draw_boxes(frame, yolo_model, existing_tracks)
            total_car_count += car_count

            out.write(frame)

        video.release()
        out.release()

        # Display the processed video
        st.write(f"Total Cars Counted: {total_car_count}")
        st.video(processed_video_path)

if __name__ == "__main__":
    main()
