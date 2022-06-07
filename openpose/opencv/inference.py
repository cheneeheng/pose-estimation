"""
Code based on:
https://learnopencv.com/opencv-dnn-with-gpu-support/
https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py
"""

import cv2
import numpy as np
import os
import time
from tqdm import trange

from openpose.opencv import PyOpenPoseOpenCV
from openpose.opencv import get_body_parts_and_pose_pairs


def save_keypoints(keypoints_list: np.ndarray, save_path: str):
    save_str = ",".join([str(i)
                         for keypoint in keypoints_list.tolist()
                         for i in keypoint])
    with open(save_path, 'a+') as f:
        f.write(f'{save_str}\n')


def save_personwise_keypoints(personwise_keypoints: list, save_path: str):
    save_str = ",".join([str(i)
                         for keypoint_idx in personwise_keypoints
                         for i in keypoint_idx.tolist()])
    with open(save_path, 'a+') as f:
        f.write(f'{save_str}\n')


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

    basepath = "/data/openpose/opencv_data/models"
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

    t_total = 0
    N = 1

    for _ in trange(100):

        image = cv2.imread(image_path)
        image = cv2.resize(image, (480, 480))

        t_start = time.time()

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

        save_keypoints(keypoints_list,
                       f'{target_path}/kptl{int(time.time() * 1e8)}.txt')

        t_total += time.time() - t_start

    print(f"Average inference time over {N} trials : {t_total/100}s")
