import cv2
import numpy as np
import torch
import time
# from playsound import playsound
from ultralytics import YOLO
import pyautogui
import pygame
# mixer.init()
pygame.mixer.init()

CONFIDENCE_THRESH = 0.6
objectClass = ["Rumble Strip", "Speed Bump"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("weights/SpeedBumpMinimBayanganTidakJauhYOLOV8sAdam.pt").to(device)
# model = YOLO("weights/yolov8s.pt")

rumble_strip_sound = pygame.mixer.Sound('Rumble Strip.mp3')
speed_bump_sound = pygame.mixer.Sound('Polisi Tidur.mp3')

# VIDEO_ROOT_PATH = "E:/Penelitian/Testing Video/"
# filename = "Project SpeedBump Test 1-3.mp4"


# cap = cv2.VideoCapture(VIDEO_ROOT_PATH + filename)
# cap = cv2.VideoCapture(1) #webcam external

cap = cv2.VideoCapture(0) #webcam internal

if not cap.isOpened():
    print("Webcam tidak terbaca")
    exit()

INPUT_VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
INPUT_VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
SCALED_WIDTH = 640
SCALED_HEIGHT = 640

# device_screen_width = 1280
# device_screen_height = 720
# device_screen_width = cv2.getWindowImageRect('Webcam YOLOv8 Object Detection')[2]
# device_screen_height = cv2.getWindowImageRect('Webcam YOLOv8 Object Detection')[3]
screen_width, screen_height = pyautogui.size()

# print(f"Testing ${filename} Video using YOLOV8 Adam CUDA")
print("Device yang digunakan: ", device)
print("SCALED_WIDTH, SCALED_HEIGHT", SCALED_WIDTH, SCALED_HEIGHT)

# Initialize variables to calculate FPS
frame_count = 0
start_time = time.time()

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# out = cv2.VideoWriter('Project SpeedBump Test 1-3 Result Adam 75 Epoch CUDA.mp4', fourcc, 30, (INPUT_VIDEO_WIDTH, INPUT_VIDEO_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    scaled_frame = cv2.resize(frame, (SCALED_WIDTH, SCALED_HEIGHT))
    # preprocessed_frame = preprocess_frame(scaled_frame)

    # print("preprocessed_frame:", preprocessed_frame)
    # outputs = model(preprocessed_frame)
    outputs = model(scaled_frame)

    for output in outputs:
        boxes = output.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # roi = preprocessed_frame[y1:y2, x1:x2]

            # Convert back to its size
            x1 = int(x1 * INPUT_VIDEO_WIDTH / SCALED_WIDTH)
            y1 = int(y1 * INPUT_VIDEO_HEIGHT / SCALED_HEIGHT)
            x2 = int(x2 * INPUT_VIDEO_WIDTH / SCALED_WIDTH)
            y2 = int(y2 * INPUT_VIDEO_HEIGHT / SCALED_HEIGHT)
            # print("roi:", roi)
            confidence = box.conf[0]
            label = box.cls[0]

            print("Confidence --->",confidence)
            print("label", label)
            print("tipe label", type(label))
            # print("Class name -->", objectClass[label])

            if confidence >= CONFIDENCE_THRESH:
                if objectClass[int(label)] == "Speed Bump":
                    speed_bump_sound.play()
                else:
                    rumble_strip_sound.play()
            # if confidence >= CONFIDENECE_THRESH and np.mean(roi) < 15:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(frame, f'{objectClass[int(label)]} {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # Increment frame count
    frame_count += 1
    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    frame_resized = cv2.resize(frame, (screen_width, screen_height))

    # Display FPS on frame
    cv2.putText(frame_resized, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # out.write(frame)
    # cv2.namedWindow('Webcam YOLOv8 Object Detection', cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty('Webcam YOLOv8 Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Webcam YOLOv8 Object Detection', frame_resized)
    # cv2.imshow('Webcam YOLOv8 Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()


