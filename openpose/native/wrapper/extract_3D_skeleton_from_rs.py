from genericpath import exists
import cv2
import json
import numpy as np
import os
from numpy.lib.stride_tricks import as_strided
import pyrealsense2 as rs

from datetime import datetime

from .skeleton import PyOpenPoseNative
from .skeleton import get_3d_skeleton
from .realsense import RealsenseWrapper


def data_storage_setup():
    date_time = datetime.now()
    date_time = date_time.strftime("%Y%m%d%H%M%S")
    save_path_calib = f'/data/{date_time}/calib'
    save_path_depth = f'/data/{date_time}/depth'
    save_path_rgb = f'/data/{date_time}/rgb'
    save_path_timestamp = f'/data/{date_time}/timestamp'
    os.makedirs(save_path_rgb, exist_ok=True)
    os.makedirs(save_path_depth, exist_ok=True)
    os.makedirs(save_path_calib, exist_ok=True)
    os.makedirs(save_path_timestamp, exist_ok=True)
    return save_path_rgb, save_path_depth, save_path_calib, save_path_timestamp


if __name__ == "__main__":

    state = True
    display_rs = False
    display_skel = True

    kpt_arr, skel_arr = None, None

    params = dict()
    params["model_folder"] = "/usr/local/src/openpose/models/"
    params["model_pose"] = "BODY_25"
    params["net_resolution"] = "-1x368"

    pyop = PyOpenPoseNative(params)
    pyop.initialize()

    sp_rgb, sp_depth, sp_calib, sp_timestamp = data_storage_setup()

    rsw = RealsenseWrapper()
    rsw.configure()
    rsw.initialize()
    rsw.save_calibration(save_path=sp_calib)

    assert os.path.exists(sp_timestamp)
    if os.path.isfile(sp_timestamp):
        timestamp_file = open(sp_timestamp, 'a')
    else:
        timestamp_file = open(os.path.join(sp_timestamp, 'timestamp.txt'), 'w')

    try:
        while state:
            state, color_image, depth_image = rsw.run(
                rgb_save_path=sp_rgb,
                depth_save_path=sp_depth,
                timestamp_file=timestamp_file,
                display=display_rs)

            # bgr format
            pyop.predict(color_image)

            scores = pyop.pose_scores
            max_score_idx = np.argmax(scores)

            keypoint = pyop.pose_keypoints[max_score_idx]
            keypoint = np.expand_dims(keypoint, axis=0)

            keypoint_image = pyop.opencv_image
            cv2.putText(keypoint_image,
                        "KP (%) : " + str(round(max(scores), 2)),
                        (10, 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA)

            if display_skel:
                cv2.imshow('keypoint_image', keypoint_image)
                key = cv2.waitKey(30)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    cv2.waitKey(5)
                    break

            # skeleton_rgb_path = rgb_file_path.replace("rgb", "skeleton_rgb")
            # skeleton_rgb_path = skeleton_rgb_path[:-4] + ".jpg"
            # cv2.imwrite(skeleton_rgb_path, keypoint_image)

            skeleton3d = get_3d_skeleton(
                keypoint[0],
                depth_image,
                rsw.calib_data['rgb']['intrinsic_mat'])

            # skeleton3d = np.expand_dims(skeleton3d, axis=0)
            # if skel_arr is None:
            #     skel_arr = np.copy(skeleton3d)
            # else:
            #     skel_arr = np.append(skel_arr, skeleton3d, axis=0)

    except:  # noqa
        print("Stopping realsense...")
        rsw.pipeline.stop()

    finally:
        rsw.pipeline.stop()
