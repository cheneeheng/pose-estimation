import argparse
import cv2
import numpy as np
import os

from datetime import datetime
from functools import partial

from skeleton import PyOpenPoseNative
from skeleton import get_3d_skeleton
from realsense import RealsenseWrapper


def data_storage_setup(device: str):
    path_dict = {}
    date_time = datetime.now().strftime("%y%m%d%H%M%S")
    path_dict['calib'] = f'/data/openpose/calib/{date_time}/dev{device}'
    path_dict['rgb'] = f'/data/openpose/rgb/{date_time}/dev{device}'
    path_dict['depth'] = f'/data/openpose/depth/{date_time}/dev{device}'
    path_dict['skeleton'] = f'/data/openpose/skeleton/{date_time}/dev{device}'
    path_dict['timestamp'] = f'/data/openpose/timestamp/{date_time}/dev{device}'
    for k, v in path_dict.items():
        os.makedirs(v, exist_ok=True)
    return path_dict


def save_skel3d(skel3d, sp_skeleton, timestamp):
    skel_file = os.path.join(sp_skeleton, f'{timestamp:020d}' + '.txt')
    skel3d_str = ",".join([str(pos)
                           for skel in skel3d.tolist()
                           for pos in skel])
    with open(skel_file, 'a+') as f:
        f.write(f'{skel3d_str}\n')

def save_skel(arg, pyop, rsw, depth_image, empty_skel3d, skeleton_sp, timestamp):
    scores = pyop.pose_scores
    max_score_idxs = np.argsort(scores)[-arg.max_true_body:]
    print(f">>>>> {timestamp:10d} : {scores[max_score_idxs[-1]]:.4f}")

    for max_score_idx in max_score_idxs:
        if scores[max_score_idx] < arg.save_skel_thres:
            save_skel3d(empty_skel3d, dev1_path_dict['skeleton'], timestamp)
            print("Low skeleton score, skip skeleton...")

        else:
            keypoint = pyop.pose_keypoints[max_score_idx]
            # ntu_format => x,y(up),z(neg) in meter.
            skel3d = get_3d_skeleton(
                skeleton=keypoint,
                depth_img=depth_image,
                intr_mat=rsw.calib_data['rgb'][0]['intrinsic_mat'],
                ntu_format=arg.ntu_format
            )
            save_skel3d(skel3d, skeleton_sp, timestamp)

    for _ in range(arg.max_true_body-len(max_score_idxs)):
        save_skel3d(empty_skel3d, skeleton_sp, timestamp)

def display_skel(pyop, scores, id=0):
    keypoint_image = pyop.opencv_image
    cv2.putText(keypoint_image,
                "KP (%) : " + str(round(max(scores), 2)),
                (10, 20),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                1,
                cv2.LINE_AA)
    cv2.imshow(f'keypoint_image_{id}', keypoint_image)


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
    parser.add_argument('--ntu-format',
                        type=bool,
                        default=False,
                        help='whether to use coordinate system of NTU')

    return parser


if __name__ == "__main__":

    device1 = '001622070408'
    device2 = '001622070717'

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
    dev1_path_dict = data_storage_setup(device1)
    dev1_ts_file = os.path.join(dev1_path_dict['timestamp'], 'timestamp.txt')
    dev2_path_dict = data_storage_setup(device2)
    dev2_ts_file = os.path.join(dev2_path_dict['timestamp'], 'timestamp.txt')

    # REALSENSE
    dev1_rsw = RealsenseWrapper()
    dev1_rsw.configure(fps=15, device=device1)
    dev1_rsw.initialize()
    dev1_rsw.save_calibration(save_path=dev1_path_dict['calib'])
    dev2_rsw = RealsenseWrapper()
    dev2_rsw.configure(fps=15, device=device2)
    dev2_rsw.initialize()
    dev2_rsw.save_calibration(save_path=dev2_path_dict['calib'])

    try:
        while state:

            # 1. Get rs data ---------------------------------------------------
            state, color_image, depth_image, timestamp = dev1_rsw.run(
                rgb_save_path=dev1_path_dict['rgb'],
                depth_save_path=dev1_path_dict['depth'],
                timestamp_file=dev1_ts_file,
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
                    save_skel3d(empty_skel3d, dev1_path_dict['skeleton'], timestamp)
                print("No skeleton detected...")
                continue

            # 3.b. Save prediction scores --------------------------------------
            if arg.save_skel:
                save_skel(arg, pyop, dev1_rsw, depth_image, 
                          empty_skel3d, dev1_path_dict['skeleton'], timestamp)

            if arg.display_skel:
                display_skel(pyop, scores, 1)
                key = cv2.waitKey(30)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    cv2.waitKey(5)
                    break

            # 1. Get rs data ---------------------------------------------------
            state, color_image, depth_image, timestamp = dev2_rsw.run(
                rgb_save_path=dev2_path_dict['rgb'],
                depth_save_path=dev2_path_dict['depth'],
                timestamp_file=dev2_ts_file,
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
                    save_skel3d(empty_skel3d, dev2_path_dict['skeleton'], timestamp)
                print("No skeleton detected...")
                continue

            # 3.b. Save prediction scores --------------------------------------
            if arg.save_skel:
                save_skel(arg, pyop, dev2_rsw, depth_image, 
                          empty_skel3d, dev2_path_dict['skeleton'], timestamp)

            if arg.display_skel:
                display_skel(pyop, scores, 2)
                key = cv2.waitKey(30)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    cv2.waitKey(5)
                    break


    except:  # noqa
        print("Stopping realsense...")
        dev1_rsw.pipeline.stop()
        dev2_rsw.pipeline.stop()

    finally:
        dev1_rsw.pipeline.stop()
        dev2_rsw.pipeline.stop()
