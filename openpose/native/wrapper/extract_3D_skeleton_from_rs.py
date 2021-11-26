import cv2
import json
import numpy as np
import os

from datetime import datetime

from skeleton import PyOpenPoseNative
from skeleton import get_3d_skeleton
from realsense import RealsenseWrapper


def data_storage_setup():
    date_time = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path_calib = f'/data/openpose/calib/{date_time}'
    save_path_rgb = f'/data/openpose/rgb/{date_time}'
    save_path_depth = f'/data/openpose/depth/{date_time}'
    save_path_skeleton = f'/data/openpose/skeleton/{date_time}'
    save_path_timestamp = f'/data/openpose/timestamp/{date_time}'
    os.makedirs(save_path_calib, exist_ok=True)
    os.makedirs(save_path_rgb, exist_ok=True)
    os.makedirs(save_path_depth, exist_ok=True)
    os.makedirs(save_path_skeleton, exist_ok=True)
    os.makedirs(save_path_timestamp, exist_ok=True)
    return (save_path_calib, save_path_rgb, save_path_depth,
            save_path_skeleton, save_path_timestamp)


if __name__ == "__main__":

    state = True
    display_rs = False
    display_skel = False

    save_skel = True

    max_true_body = 2

    params = dict()
    params["model_folder"] = "/usr/local/src/openpose/models/"
    params["model_pose"] = "BODY_25"
    params["net_resolution"] = "-1x368"

    pyop = PyOpenPoseNative(params)
    pyop.initialize()

    sp_calib, sp_rgb, sp_depth, sp_skeleton, sp_ts = data_storage_setup()

    rsw = RealsenseWrapper()
    rsw.configure()
    rsw.initialize()
    rsw.save_calibration(save_path=sp_calib)

    timestamp_file = os.path.join(sp_ts, 'timestamp.txt')

    try:
        while state:

            # 1. Get rs data ---------------------------------------------------
            state, color_image, depth_image, timestamp = rsw.run(
                rgb_save_path=sp_rgb,
                depth_save_path=sp_depth,
                timestamp_file=timestamp_file,
                display=display_rs)

            # 2. Predict pose --------------------------------------------------
            # bgr format
            pyop.predict(color_image)

            # 3. Get prediction scores -----------------------------------------
            scores = pyop.pose_scores

            # 3.a. Save empty array if scores is None (no skeleton at all) -----
            if scores is None:
                skeleton3d = np.zeros((25, 3))
                skel_file = os.path.join(sp_skeleton, f'{timestamp}' + '.txt')
                skeleton3d_str = ",".join([str(pos)
                                           for skel in skeleton3d.tolist()
                                           for pos in skel])
                with open(skel_file, 'a+') as f:
                    f.write(f'{skeleton3d_str}')
                    continue

            # 3.b. Save prediction scores --------------------------------------
            # max_score_idx = np.argmax(scores)
            max_score_idxs = np.argsort(scores)[-max_true_body:]
            print(f"{timestamp:d} : {scores[max_score_idxs[-1]]:.4f}")

            if save_skel:
                for max_score_idx in max_score_idxs:
                    keypoint = pyop.pose_keypoints[max_score_idx]
                    skeleton3d = get_3d_skeleton(
                        keypoint,
                        depth_image,
                        rsw.calib_data['rgb']['intrinsic_mat'])

                    skel_file = os.path.join(sp_ts, f'{timestamp}' + '.txt')
                    skeleton3d_str = ",".join([str(pos)
                                               for skel in skeleton3d.tolist()
                                               for pos in skel])
                    with open(skel_file, 'a+') as f:
                        f.write(f'{skeleton3d_str}')

            if display_skel:
                keypoint_image = pyop.opencv_image
                cv2.putText(keypoint_image,
                            "KP (%) : " + str(round(max(scores), 2)),
                            (10, 20),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA)
                cv2.imshow('keypoint_image', keypoint_image)
                key = cv2.waitKey(30)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    cv2.waitKey(5)
                    break

    except:  # noqa
        print("Stopping realsense...")
        rsw.pipeline.stop()

    finally:
        rsw.pipeline.stop()
