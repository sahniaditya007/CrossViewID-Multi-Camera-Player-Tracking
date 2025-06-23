import torch
from ultralytics import YOLO
import cv2

def run_detection(video_path, model_path, device='cpu'):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = model(frame)[0]

        frame_detections = []
        for box in results.boxes:
            cls = int(box.cls)
            if cls == 0:  # Assuming class 0 = player
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                frame_detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf
                })
        detections.append(frame_detections)
    cap.release()
    return detections
