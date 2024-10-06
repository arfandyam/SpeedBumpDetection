import numpy as np
import time
import cv2
import torch
import torchvision
import pyautogui
import pygame
pygame.mixer.init()


# As the data directory contains .xml files
from xml.etree import ElementTree as et

import warnings
warnings.filterwarnings('ignore')

def apply_nms(prediction, threshold):
    # torchvision returns the indices of the boxes to keep
    keep = torchvision.ops.nms(prediction['boxes'], prediction['scores'], threshold)

    final_prediction = prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction

# fast_rcnn = torch.load("speedbump_fastrcnn_training_7.pth")
fast_rcnn = torch.load("speedbump_fastrcnn_training_adam_minbayangan_tidakjauh_224.pth")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

rumble_strip_sound = pygame.mixer.Sound('Rumble Strip.mp3')
speed_bump_sound = pygame.mixer.Sound('Polisi Tidur.mp3')

CONFIDENCE_THRESH = 0.9
objectClass = ["background", "Speed Bump", "Rumble Strip"]

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Webcam tidak terbaca")
    exit()

INPUT_VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
INPUT_VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
SCALED_WIDTH = 224
SCALED_HEIGHT = 224

screen_width, screen_height = pyautogui.size()

print("Device yang digunakan: ", device)
print("SCALED_WIDTH, SCALED_HEIGHT", SCALED_WIDTH, SCALED_HEIGHT)

frame_count = 0
start_time = time.time()

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        scaled_frame = cv2.resize(rgb_frame, (SCALED_WIDTH, SCALED_HEIGHT))
        scaled_frame /= 255.0
        scaled_frame = np.moveaxis(scaled_frame, 2, 0) # to [C, H, W]

        # Conversion to Tensors
        scaled_frame = torch.as_tensor(scaled_frame, dtype=torch.float32)

        prediction = fast_rcnn([scaled_frame.to(device)])[0]
        nms_prediction = apply_nms(prediction, threshold=0.5)
        prediction_scores = nms_prediction['scores']

        boxes = nms_prediction['boxes'].cpu()
        scores = nms_prediction['scores'].cpu()
        labels = nms_prediction['labels'].cpu()

        # for box in boxes:
        for idx, box in enumerate(boxes):
            # x1, y1, x2, y2 = box.xyxy[0]
            # x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
            x1, y1, x2, y2  = box[0], box[1], box[2], box[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Convert back to its size
            x1 = int(x1 * INPUT_VIDEO_WIDTH / SCALED_WIDTH)
            y1 = int(y1 * INPUT_VIDEO_HEIGHT / SCALED_HEIGHT)
            x2 = int(x2 * INPUT_VIDEO_WIDTH / SCALED_WIDTH)
            y2 = int(y2 * INPUT_VIDEO_HEIGHT / SCALED_HEIGHT)
            # print("roi:", roi)
            # confidence = box.conf[0]
            confidence = scores[idx]
            label = labels[idx]
            # label = box.cls[0]

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
        cv2.imshow('Webcam Faster-RCNN Speed Bump Detection', frame_resized)
        # cv2.imshow('Webcam Faster-RCNN Speed Bump Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()