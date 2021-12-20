import argparse
import cv2
import numpy as np
import os

from datetime import datetime
from functools import partial

from skeleton import PyOpenPoseNative
from skeleton import get_3d_skeleton
from realsense import RealsenseWrapper


def data_storage_setup():
    date_time = datetime.now().strftime("%y%m%d%H%M%S")
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


def save_skel3d(skel3d, sp_skeleton, timestamp):
    skel_file = os.path.join(sp_skeleton, f'{timestamp}' + '.txt')
    skel3d_str = ",".join([str(pos)
                           for skel in skel3d.tolist()
                           for pos in skel])
    with open(skel_file, 'a+') as f:
        f.write(f'{skel3d_str}\n')


def get_parser():
    parser = argparse.ArgumentParser(
        description='Extract 3D skeleton using OPENPOSE')

    parser.add_argument('--display-rs',
                        type=bool,
                        default=False,
                        help='if true, display realsense raw images.')
    parser.add_argument('--display-skel',
                        type=bool,
                        default=False,
                        help='if true, display skel images from openpose.')

    parser.add_argument('--save-skel',
                        default=True,
                        help='if true, save 3d skeletons.')
    parser.add_argument('--save-skel-thres',
                        type=float,
                        default=0.5,
                        help='threshold for valid skeleton.')
    parser.add_argument('--max-true-body',
                        type=int,
                        default=2,
                        help='max number of skeletons to save.')

    parser.add_argument('--model-folder',
                        type=str,
                        default="/usr/local/src/openpose/models/",
                        help='foilder with trained openpose models.')
    parser.add_argument('--model-pose',
                        type=str,
                        default="BODY_25",
                        help=' ')
    parser.add_argument('--net-resolution',
                        type=str,
                        default="-1x368",
                        help='resolution of input to openpose.')

    return parser


if __name__ == "__main__":

    parser = get_parser()
    arg = parser.parse_args()

    state = True
    empty_skel3d = np.zeros((25, 3))

    # 0. Initialize ------------------------------------------------------------
    # OPENPOSE
    params = dict(
        model_folder=arg.model_folder,
        model_pose=arg.model_pose,
        net_resolution=arg.net_resolution,
    )
    pyop = PyOpenPoseNative(params)
    pyop.initialize()

    # STORAGE
    sp_calib, sp_rgb, sp_depth, sp_skeleton, sp_ts = data_storage_setup()
    timestamp_file = os.path.join(sp_ts, 'timestamp.txt')

    # REALSENSE
    rsw = RealsenseWrapper()
    rsw.configure(fps=15)
    rsw.initialize()
    rsw.save_calibration(save_path=sp_calib)

    try:
        while state:

            # 1. Get rs data ---------------------------------------------------
            state, color_image, depth_image, timestamp = rsw.run(
                rgb_save_path=sp_rgb,
                depth_save_path=sp_depth,
                timestamp_file=timestamp_file,
                display=arg.display_rs
            )

            # 2. Predict pose --------------------------------------------------
            # bgr format
            pyop.predict(color_image)

            # 3. Get prediction scores -----------------------------------------
            scores = pyop.pose_scores

            # 3.a. Save empty array if scores is None (no skeleton at all) -----
            if scores is None:
                for _ in range(arg.max_true_body):
                    save_skel3d(empty_skel3d, sp_skeleton, timestamp)
                print("No skeleton detected...")
                continue

            # 3.b. Save prediction scores --------------------------------------
            # max_score_idx = np.argmax(scores)
            max_score_idxs = np.argsort(scores)[-arg.max_true_body:]
            print(f">>>>> {timestamp:10d} : {scores[max_score_idxs[-1]]:.4f}")

            if arg.save_skel:
                for max_score_idx in max_score_idxs:

                    if scores[max_score_idx] < arg.save_skel_thres:
                        save_skel3d(empty_skel3d, sp_skeleton, timestamp)
                        print("Low skeleton score, skip skeleton...")

                    else:
                        keypoint = pyop.pose_keypoints[max_score_idx]
                        skel3d = get_3d_skeleton(
                            skeleton=keypoint,
                            depth_img=depth_image,
                            intr_mat=rsw.calib_data['rgb'][0]['intrinsic_mat']
                        )
                        save_skel3d(skel3d, sp_skeleton, timestamp)

                for _ in range(arg.max_true_body-len(max_score_idxs)):
                    save_skel3d(empty_skel3d, sp_skeleton, timestamp)

            if arg.display_skel:
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
