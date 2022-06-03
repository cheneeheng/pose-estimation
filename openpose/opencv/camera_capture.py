import cv2
import numpy as np
import time

import os
import sys
sys.path.append(os.getcwd())

from openpose.opencv.inference import get_body_parts_and_pose_pairs  # noqa
from openpose.opencv.inference import PyOpenPose  # noqa


# -----------------------------------------------------------------------------
basepath = "/home/chen/data/03_OpenPose/models"
protoFile = basepath + "/pose/coco/pose_deploy_linevec.prototxt"
weightsFile = basepath + "/pose/coco/pose_iter_440000.caffemodel"
dataset = 'COCO'

conf_thres = 0.2

image_width = 368*2
image_height = 368

body_parts, pose_pairs = get_body_parts_and_pose_pairs(dataset)

# Openpose model wrapper.
op = PyOpenPose(image_width=image_width,
                image_height=image_height,
                body_parts=body_parts,
                pose_pairs=pose_pairs,
                conf_thres=conf_thres)
op.load(protoFile, weightsFile)

# -----------------------------------------------------------------------------
# cap = cv2.VideoCapture("udp://172.29.14.125:5001")
cap = cv2.VideoCapture('udp://127.0.0.1:5001', cv2.CAP_FFMPEG)

if not cap.isOpened():
    print('VideoCapture not opened')
    exit(-1)

downsample_factor = 6
count = 0

while True:

    ret, ori_frame = cap.read()

    if not ret:
        print('frame empty')
        break

    if count < downsample_factor:
        count += 1
        continue
    else:
        count = 0

    frame = cv2.resize(ori_frame, (image_width, image_height))
    # frame = ori_frame

    pred = op.predict(frame)
    # op.postprocess_single(pred, image, output_image_path="opencv/output/tmp.png")  # noqa
    _, _, image = op.postprocess_multi(pred, frame)
    # _, image = op.postprocess_single(pred, frame)

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    cv2.putText(image, current_time, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.imshow('image', image)

    # cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

    del frame

cap.release()
cv2.destroyAllWindows()
