import cv2
import json
import numpy as np
import os
import pyopenpose as op
import sys

from tqdm import tqdm
from time import sleep

from .skeleton import PyOpenPoseNative
from .skeleton import get_3d_skeleton
from .realsense import read_realsense_calibration


if __name__ == "__main__":

    CLIPS_PATH = "/DigitalICU/ICRA2022/data/all/clips"

    params = dict()
    params["model_folder"] = "/usr/local/src/openpose/models/"
    params["model_pose"] = "BODY_25"
    params["net_resolution"] = "-1x368"

    pyop = PyOpenPoseNative(params)
    pyop.initialize()

    for clip_id in sorted(os.listdir(CLIPS_PATH))[int(sys.argv[1]):int(sys.argv[2])]:  # noqa
        # for clip_id in sorted(os.listdir(CLIPS_PATH))[549:550]:
        clip_path = os.path.join(CLIPS_PATH, clip_id)

        print("Processing :", clip_path)

        calib_path = os.path.join(clip_path, 'calib.txt')
        calib_data = read_realsense_calibration(calib_path)

        rgb_path = os.path.join(clip_path, 'rgb')
        depth_path = os.path.join(clip_path, 'depth')

        os.makedirs(os.path.join(clip_path, 'skeleton_rgb'), exist_ok=True)
        assert not os.listdir(os.path.join(clip_path, 'skeleton_rgb'))

        kpt_arr, skel_arr = None, None

        for rgb_file in tqdm(sorted(os.listdir(rgb_path))):

            rgb_file_path = os.path.join(rgb_path, rgb_file)
            # rgb_file_path = '/home/chen/openpose/pexels-photo-4384679.jpeg'
            rgb_img = cv2.imread(rgb_file_path)
            # rgb_img = cv2.resize(rgb_img, (368, 368))

            pyop.predict(rgb_img)

            scores = pyop.pose_scores
            max_score_idx = np.argmax(scores)

            keypoint = pyop.pose_keypoints[max_score_idx]
            keypoint = np.expand_dims(keypoint, axis=0)
            if kpt_arr is None:
                kpt_arr = np.copy(keypoint)
            else:
                kpt_arr = np.append(kpt_arr, keypoint, axis=0)

            keypoint_image = pyop.opencv_image
            cv2.putText(keypoint_image,
                        "KP (%) : " + str(round(max(scores), 2)),
                        (10, 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA)
            skeleton_rgb_path = rgb_file_path.replace("rgb", "skeleton_rgb")
            skeleton_rgb_path = skeleton_rgb_path[:-4] + ".jpg"
            cv2.imwrite(skeleton_rgb_path, keypoint_image)

            depth_file_path = os.path.join(depth_path, rgb_file)
            depth_img = cv2.imread(depth_file_path, -1)

            skeleton3d = get_3d_skeleton(keypoint[0],
                                         depth_img,
                                         calib_data.rgb_intrinsics)
            skeleton3d = np.expand_dims(skeleton3d, axis=0)
            if skel_arr is None:
                skel_arr = np.copy(skeleton3d)
            else:
                skel_arr = np.append(skel_arr, skeleton3d, axis=0)

            # print("Skeleton 3D :", skeleton3d)

            sleep(0.01)

        npy_path = os.path.join(clip_path, 'skeleton.npy')
        np.save(npy_path, kpt_arr)
        npy_path = os.path.join(clip_path, 'skeleton_3d.npy')
        np.save(npy_path, skel_arr)

    # Display Image
    # print("Body keypoints: \n" + str(pyop.pose_keypoints))
    # print("Prediction score: \n" + str(pyop.pose_scores))
    # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", keypoint_image)
    # cv2.waitKey(0)
