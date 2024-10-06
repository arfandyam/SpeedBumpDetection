import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.models.detection import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# from .vision.references.detection import transforms as T, utils
# from .vision.references.detection.engine import train_one_epoch, evaluate
# from engine import train_one_epoch, evaluate
# import utils
# import transforms as T

# For image augmentations
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2

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

# Function to convert a torch tensor to a PIL Image
# def tensorToPIL(img):
    return transforms.ToPILImage()(img).convert('RGB')

# class SpeedBumpImagetTestDataset(torch.utils.data.Dataset):
    def __init__(self, files_dir, width, height, transforms=None):
        self.files_dir = files_dir
        self.width = width
        self.height = height
        self.transforms = transforms  # If transformation is required, when transforms is not None

        self.classes_ = [_, 'Speed-Bump', 'Rumble Strip']   # Defining classes, a blank class is given for the background

        self.images = [img for img in sorted(os.listdir(files_dir)) if img[-4:]=='.jpg']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.files_dir, img_name)

        # Reading the image
        img = cv2.imread(img_path)
        # img = Image.open(img_path)

        # Defining width and height
        # wt, ht = img.size
        wt = img.shape[1]
        ht = img.shape[0]

        # Converting image to RGB channel and normalizing the image
        # img = cv2.resize(np.array(img), (self.width, self.height), cv2.INTER_AREA).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (self.width, self.height), cv2.INTER_AREA)
        img /= 255.0

        annot_name = img_name[:-4] + '.xml'
        annot_path = os.path.join(self.files_dir, annot_name)

        # Boxes to store the coordinate points of the bboxes
        boxes, labels = [], []

        tree = et.parse(annot_path)
        root = tree.getroot()

        # Box coordinates are extracted from the XML files for the given image size
        for member in root.findall('object'):
            labels.append(self.classes_.index(member.find('name').text))

            xmin = float(member.find('bndbox').find('xmin').text)
            xmax = float(member.find('bndbox').find('xmax').text)
            ymin = float(member.find('bndbox').find('ymin').text)
            ymax = float(member.find('bndbox').find('ymax').text)

            x_min = (xmin/wt)*self.width
            x_max = (xmax/wt)*self.width
            y_min = (ymin/ht)*self.height
            y_max = (ymax/ht)*self.height

            boxes.append([x_min, y_min, x_max, y_max])
        
        img = np.moveaxis(img, 2, 0) # to [C, H, W]
        # Conversion to Tensors
        img = torch.as_tensor(img, dtype=torch.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # Calculating area of the boxes

        iscrowd = torch.zeros((boxes.shape[0], ), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
#         image_id = idx

        target = {'boxes': boxes, 'area': area, 'labels': labels,
                'iscrowd': iscrowd, 'image_id':image_id}

        if self.transforms:
            sample = self.transforms(image = img,
                                    bboxes = target['boxes'],
                                    labels = labels)

            img = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return img, target
    
# def plot_img_bbox_2(img, target, score_thresh):

    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    print(target)
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)
    
    boxes = target['boxes'].cpu()
    scores = target['scores'].cpu()

    for idx, box in enumerate(boxes):
        if scores[idx] > score_thresh:
            x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
            rect = patches.Rectangle((x, y),
                                     width, height,
                                     linewidth = 2,
                                     edgecolor = 'r',
                                     facecolor = 'none')

            # Draw the bounding box on top of the image
            a.add_patch(rect)
    plt.show()

fast_rcnn = torch.load("speedbump_fastrcnn_training_1.pth")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("DEVICE YANG DIGUNAKAN: ", device)
CONFIDENCE_THRESH = 0.9
objectClass = ["background", "Speed Bump", "Rumble Strip"]

VIDEO_ROOT_PATH = "E:/Penelitian/Testing Video/"
filename = "Project SpeedBump Test 1.mp4"


cap = cv2.VideoCapture(VIDEO_ROOT_PATH + filename)

INPUT_VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
INPUT_VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
SCALED_WIDTH = 480
SCALED_HEIGHT = 480

print(f"Testing ${filename} Video using FasterRCNN")
print("SCALED_WIDTH, SCALED_HEIGHT", SCALED_WIDTH, SCALED_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('Project SpeedBump Test 1 Result FasterRCNN SGD 25 Epoch.mp4', fourcc, 30, (INPUT_VIDEO_WIDTH, INPUT_VIDEO_HEIGHT))

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
            # if confidence >= CONFIDENECE_THRESH and np.mean(roi) < 15:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(frame, f'{objectClass[int(label)]} {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)


        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()