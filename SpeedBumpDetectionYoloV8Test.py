import cv2
import numpy as np
import torch
# from playsound import playsound
from ultralytics import YOLO

CONFIDENCE_THRESH = 0.6
objectClass = ["Rumble Strip", "Speed Bump"]

# def preprocess_frame(frame):
#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
#     # Apply adaptive thresholding to grayscale image
#     # adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     #                                         cv2.THRESH_BINARY_INV, 11, 2)
#     _, binary_gray = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    
#     # Define HSV range for dark colors (adjust these values as needed)
#     lower_dark = np.array([0, 0, 0])
#     # upper_dark = np.array([50, 50, 50])
#     upper_dark = np.array([180, 255, 150])
    
#     # Create a mask for dark colors
#     mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
#     # Combine the two masks
#     combined_mask = cv2.bitwise_and(binary_gray, mask_dark)
#     # combined_mask = cv2.bitwise_and(adaptive_thresh, mask_dark)
    
#     # Apply morphological operations to reduce noise
#     kernel = np.ones((3, 3), np.uint8)
#     combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
#     combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)


#     # Apply histogram equalization
#     # equalized = cv2.equalizeHist(gray)
#     # # Convert back to BGR
#     # preprocessed_frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
#     # return preprocessed_frame
#     return combined_mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("weights/best_adam_75epoch.pt").to(device)
# model = YOLO("weights/yolov8s.pt")

VIDEO_ROOT_PATH = "E:/Penelitian/Testing Video/"
filename = "Project SpeedBump Test 1-3.mp4"


cap = cv2.VideoCapture(VIDEO_ROOT_PATH + filename)

INPUT_VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
INPUT_VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
SCALED_WIDTH = 640
SCALED_HEIGHT = 640

print(f"Testing ${filename} Video using YOLOV8 Adam CUDA")
print("Device yang digunakan: ", device)
print("SCALED_WIDTH, SCALED_HEIGHT", SCALED_WIDTH, SCALED_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('Project SpeedBump Test 1-3 Result Adam 75 Epoch CUDA.mp4', fourcc, 30, (INPUT_VIDEO_WIDTH, INPUT_VIDEO_HEIGHT))

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
                # if(int(label)):
                #     playsound('Polisi Tidur.mp3')
                # else:
                #     playsound('Rumble Strip.mp3')
            # if confidence >= CONFIDENECE_THRESH and np.mean(roi) < 15:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(frame, f'{objectClass[int(label)]} {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


