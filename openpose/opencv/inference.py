"""
Code based on:
https://learnopencv.com/opencv-dnn-with-gpu-support/
https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py
"""

import cv2
import os

from openpose.opencv import PyOpenPoseOpenCV
from openpose.opencv import get_body_parts_and_pose_pairs


if __name__ == "__main__":

    target_path = "openpose/output/inference_opencv"
    os.makedirs(target_path, exist_ok=True)

    image_height = 368*1
    image_width = 368*1

    image_path = "openpose/pexels-photo-4384679.jpeg"
    # image = cv2.imread(image_path)
    # image = cv2.resize(image, (368, 368))
    # cv2.imshow('OpenPose using OpenCV', image)
    # cv2.waitKey(0)

    basepath = "/home/chen/data/03_OpenPose/models"
    protoFile = basepath + "/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = basepath + "/pose/coco/pose_iter_440000.caffemodel"
    dataset = 'COCO'

    # basepath = "/home/chen/data/03_OpenPose/models"
    # protoFile = basepath + "/pose/mpi/pose_deploy_linevec.prototxt"
    # weightsFile = basepath + "/pose/mpi/pose_iter_160000.caffemodel"
    # dataset = 'MPI'

    body_parts, pose_pairs = get_body_parts_and_pose_pairs(dataset)
    pyop = PyOpenPoseOpenCV(image_height, image_width, body_parts, pose_pairs)
    pyop.load(protoFile, weightsFile, False)

    for _ in range(100):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (848, 480))
        pred = pyop.predict(image)
        # points = pyop.postprocess_single(pred)
        # pyop.draw_skeleton(points=points,
        #                    image=image,
        #                    show_image=False,
        #                    output_image_path="opencv/output/tmp.png")
        keypoints_list, personwise_keypoints = pyop.postprocess_multi(pred)
        # pyop.draw_skeleton(personwise_keypoints=personwise_keypoints,
        #                    keypoints_list=keypoints_list,
        #                    image=image,
        #                    show_image=False,
        #                    output_image_path="opencv/output/tmp.png")
