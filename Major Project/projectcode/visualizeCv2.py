import cv2 as cv
import cv2
import numpy as np
import os
import sys
# from twilio.rest import TwilioRestClient as Call
from mrcnn import utils
from mrcnn import model as modellib
from tkinter import * 
from tkinter import messagebox 
import playsound
import easygui    
from datetime import datetime
import string
import random
# from alert import call


ROOT_DIR = os.path.abspath("../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import pycocotools
import coco
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)
}


def apply_mask(image, mask, color, alpha=0.5):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image



def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image* measured_distance)/ real_width
    return focal_length
# distance estimation function

def Distance_finder (Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length)/face_width_in_frame
    return distance

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# For Width of the frame
def face_data(image):
    face_width = 0 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (222,141,123), 1)
        face_width = w

    return face_width

previous_distance = 30 

width =14.3 


# To compare the distance we need another image 
img = cv2.imread("rf.png")
face_width = face_data(img)
Focal_length = FocalLength(previous_distance, width,face_width)

video = cv2.VideoCapture(0)
while True:
    ret, frame = video.read()
    if ret==True:
        face_width_in_frame = face_data(frame)
        if face_width_in_frame !=0:
            Distance = round(Distance_finder(Focal_length, width,face_width_in_frame),2)
            cv2.putText(frame, "Distance from Camera "+"{}".format(Distance)+"CM", (50,50), cv2.FONT_HERSHEY_COMPLEX,1, (123,246,123),3)
            if(Distance>30):
                easygui.msgbox("People found outside the border!!!", title="Alert")
            else:
                easygui.msgbox("People intruded the border!!!", title="Alert")
        cv2.imshow("frame", frame )
        if cv2.waitKey(1)==ord("q"):
            break 
video.release()
cv2.destroyAllWindows()






def display_instances(image, boxes, masks, ids, names, scores):
    n_instances = boxes.shape[0]


    # if not n_instances:
    #     print('NO INSTANCES TO DISPLAY')
    # else:
    #     assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    # elephant = 0
    # for i in range(n_instances):
    #     if not np.any(boxes[i]):
    #         continue
    #     y1, x1, y2, x2 = boxes[i]
    #     label = names[ids[i]]
    #     if(label != "elephant"):
    #         continue
    #     elephant = 1
    #     color = class_dict[label]
    #     score = scores[i] if scores is not None else None
    #     caption = '{} {:.2f}'.format(label, score) if score else label
    #     mask = masks[:, :, i]

    #     image = apply_mask(image, mask, color)
    #     image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    #     image = cv2.putText(
    #         image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
    #     )
    # if(elephant == 1):
    #     N = 4
    #     res = ''.join(random.choices(string.ascii_uppercase +string.digits, k = N))
    #     now = datetime.now()
    #     date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    #     color = class_dict[label]
    #     img = cv2.putText(
    #         image, date_time, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
    #     )
    #     cv2.imwrite('./logs/frame'+res+'.jpg', img)
    #     # call()
    #     easygui.msgbox("Elephants found at the border!!!", title="Alert")



    # if not n_instances:
    #         print('NO INSTANCES TO DISPLAY')
    # else:
    #     assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    # sheep = 0
    # for i in range(n_instances):
    #     if not np.any(boxes[i]):
    #         continue
    #     y1, x1, y2, x2 = boxes[i]
    #     label = names[ids[i]]
    #     if(label != "sheep"):
    #         continue
    #     sheep = 1
    #     color = class_dict[label]
    #     score = scores[i] if scores is not None else None
    #     caption = '{} {:.2f}'.format(label, score) if score else label
    #     mask = masks[:, :, i]

    #     image = apply_mask(image, mask, color)
    #     image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    #     image = cv2.putText(
    #         image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
    #     )
    # if(sheep == 1):
    #     N = 4
    #     res = ''.join(random.choices(string.ascii_uppercase +string.digits, k = N))
    #     now = datetime.now()
    #     date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    #     color = class_dict[label]
    #     img = cv2.putText(
    #         image, date_time, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
    #     )
    #     cv2.imwrite('./logs/frame'+res+'.jpg', img)
    #     # call()
    #     easygui.msgbox("Sheeps found at the border!!!", title="Alert")




    # if not n_instances:
    #         print('NO INSTANCES TO DISPLAY')
    # else:
    #     assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    # giraffe = 0
    # for i in range(n_instances):
    #     if not np.any(boxes[i]):
    #         continue
    #     y1, x1, y2, x2 = boxes[i]
    #     label = names[ids[i]]
    #     if(label != "giraffe"):
    #         continue
    #     giraffe = 1
    #     color = class_dict[label]
    #     score = scores[i] if scores is not None else None
    #     caption = '{} {:.2f}'.format(label, score) if score else label
    #     mask = masks[:, :, i]

    #     image = apply_mask(image, mask, color)
    #     image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    #     image = cv2.putText(
    #         image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
    #     )
    # if(giraffe == 1):
    #     N = 4
    #     res = ''.join(random.choices(string.ascii_uppercase +string.digits, k = N))
    #     now = datetime.now()
    #     date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    #     color = class_dict[label]
    #     img = cv2.putText(
    #         image, date_time, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
    #     )
    #     cv2.imwrite('./logs/frame'+res+'.jpg', img)
    #     # call()
    #     easygui.msgbox("Giraffes found at the border!!!", title="Alert")


    if not n_instances:
            print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    person = 0
    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        if(label != "person"):
            continue
        person = 1
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )
    if(person == 1):
        N = 4
        res = ''.join(random.choices(string.ascii_uppercase +string.digits, k = N))
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        color = class_dict[label]
        img = cv2.putText(
            image, date_time, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )
        cv2.imwrite('./logs/frame'+res+'.jpg', img)
        # call()
        easygui.msgbox("People found at the border!!!", title="Alert")

    return image
