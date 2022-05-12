import argparse
import cv2
import numpy as np
import os

from datetime import datetime

from skeleton import PyOpenPoseNative
from skeleton import get_3d_skeleton
from realsense import RealsenseWrapper

from infer.inference import parse_arg
from infer.inference import get_parser as get_agcn_parser
from infer.inference import init_file_and_folders
from infer.inference import init_preprocessor
from infer.inference import init_model
from infer.inference import model_inference
from infer.inference import filter_logits


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
    skel_file = os.path.join(sp_skeleton, f'{timestamp:020d}' + '.txt')
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
                        default=0.3,
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

    [op_arg, _] = get_parser().parse_known_args()
    [agcn_arg, _] = parse_arg(get_agcn_parser())

    # 0. Initialize ------------------------------------------------------------

    # STORAGE
    sp_calib, sp_rgb, sp_depth, sp_skeleton, sp_ts = data_storage_setup()
    timestamp_file = os.path.join(sp_ts, 'timestamp.txt')

    # AAGCN
    MAPPING, _, output_dir = init_file_and_folders(agcn_arg)
    DataProc = init_preprocessor(agcn_arg)
    Model = init_model(agcn_arg)

    # OPENPOSE
    params = dict(
        model_folder=op_arg.model_folder,
        model_pose=op_arg.model_pose,
        net_resolution=op_arg.net_resolution,
        disable_blending=True
    )
    pyop = PyOpenPoseNative(params)
    pyop.initialize()

    # REALSENSE
    rsw = RealsenseWrapper()
    rsw.configure(fps=30)
    rsw.initialize()
    rsw.save_calibration(save_path=sp_calib)

    state = True
    empty_skel3d = np.zeros((25, 3))

    mva = 5
    mva_list = []

    try:
        while state:

            # 1. Get rs data ---------------------------------------------------
            state, color_image, depth_image, timestamp = rsw.run(
                rgb_save_path=sp_rgb,
                depth_save_path=sp_depth,
                timestamp_file=timestamp_file,
                display=op_arg.display_rs
            )

            # 2. Predict pose --------------------------------------------------
            # bgr format
            pyop.predict(color_image)

            # 3. Get prediction scores -----------------------------------------
            scores = pyop.pose_scores

            # 3.a. Save empty array if scores is None (no skeleton at all) -----
            if scores is None:
                for _ in range(op_arg.max_true_body):
                    save_skel3d(empty_skel3d, sp_skeleton, timestamp)
                print("No skeleton detected...")
                DataProc.append_data(np.zeros((2, 1, agcn_arg.num_joint, 3)))
                continue

            # 3.b. Save prediction scores --------------------------------------
            # max_score_idx = np.argmax(scores)
            max_score_idxs = np.argsort(scores)[-op_arg.max_true_body:]
            try:
                print(f">>>>> {timestamp:10d} : {max_score_idxs} : {scores[max_score_idxs[1]]:.4f} -- {scores[max_score_idxs[0]]:.4f}")   # noqa
            except IndexError:
                print(f">>>>> {timestamp:10d} : {max_score_idxs} : {scores[max_score_idxs[0]]:.4f}")   # noqa
            except Exception as e: 
                print(e)

            skel_data = []

            for max_score_idx in max_score_idxs:
                if scores[max_score_idx] < op_arg.save_skel_thres:
                    save_skel3d(empty_skel3d, sp_skeleton, timestamp)
                    print("Low skeleton score, skip skeleton...")
                    skel_data.append(empty_skel3d)

                else:
                    keypoint = pyop.pose_keypoints[max_score_idx]
                    # ntu_format => x,y(up),z(neg) in meter.
                    skel3d = get_3d_skeleton(
                        skeleton=keypoint,
                        depth_img=depth_image,
                        intr_mat=rsw.calib_data['rgb'][0]['intrinsic_mat'],
                        ntu_format=op_arg.ntu_format
                    )
                    save_skel3d(skel3d, sp_skeleton, timestamp)
                    skel_data.append(skel3d)

            for _ in range(op_arg.max_true_body-len(max_score_idxs)):
                save_skel3d(empty_skel3d, sp_skeleton, timestamp)
                skel_data.append(empty_skel3d)

            if op_arg.display_skel:

                keypoint_image = pyop.opencv_image
                keypoint_image = cv2.flip(keypoint_image, 1)
                cv2.putText(keypoint_image,
                            "KP (%) : " + str(round(max(scores), 2)),
                            (10, 20),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA)
                # keypoint_image = cv2.resize(keypoint_image, (1280, 720))
                cv2.namedWindow('keypoint_image', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('keypoint_image',
                                     cv2.WND_PROP_FULLSCREEN,
                                     cv2.WINDOW_FULLSCREEN)
                cv2.imshow('keypoint_image', keypoint_image)

                # depth_image = np.clip(depth_image, 0, 10000)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.065, beta=0),
                    cv2.COLORMAP_INFERNO)
                depth_colormap = cv2.flip(depth_colormap, 1)
                depth_keypoint_overlay = cv2.addWeighted(
                    keypoint_image, 0.7, depth_colormap, 0.7, 0)
                depth_keypoint_overlay = cv2.resize(
                    depth_keypoint_overlay, (800, 450))
                cv2.imshow("depth_keypoint_overlay", depth_keypoint_overlay)
                cv2.moveWindow("depth_keypoint_overlay", 1500,300);

                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    cv2.waitKey(5)
                    break

            # 4. Action recognition --------------------------------------------
            # 4.1. Batch frames to fixed length.
            skel_data = np.stack(skel_data, axis=0)  # m,v,c
            skel_data = np.expand_dims(
                skel_data[:, :agcn_arg.num_joint, :],
                axis=1
            )  # m,t,v,c
            DataProc.append_data(skel_data)
            try:
                input_data = DataProc.select_skeletons_and_normalize_data(
                    agcn_arg.max_num_skeleton_true,
                    sgn='sgn' in agcn_arg.model
                )
                input_data = DataProc.data

                # 4.2. repeat segments in a sequence.
                if 'aagcn' in agcn_arg.model:
                    if np.sum(input_data) == np.nan:
                        DataProc.clear_data_array()
                        input_data = DataProc.data
                    # N, C, T, V, M
                    input_data = np.concatenate(
                        [input_data, input_data, input_data], axis=2)

                # 4.3. Inference.
                logits, preds = model_inference(agcn_arg, Model, input_data)  # noqa
                logits = logits[0]

                if len(mva_list) < mva:
                    mva_list.append(logits.numpy())
                else:
                    mva_list = mva_list[1:] + [logits.numpy()]
                    logits = np.mean(np.stack(mva_list, axis=0), axis=0)
                    mva_list[-1] = logits

                logits, preds = logits.tolist(), preds.item()
                sort_idx, new_logits = filter_logits(logits)

                output_file = os.path.join(
                    output_dir, f'{timestamp:020d}' + '.txt')
                with open(output_file, 'a+') as f:
                    output_str1 = ",".join([str(i) for i in sort_idx])
                    output_str2 = ",".join([str(i) for i in new_logits])
                    output_str = f'{output_str1};{output_str2}\n'
                    output_str = output_str.replace('[', '').replace(']', '')
                    f.write(output_str)
                    if len(sort_idx) > 0:
                        print(f"Original Pred: {preds}, Filtered Pred: {sort_idx[0]: >2}, Logit: {new_logits[0]*100:>5.2f}")  # noqa
                    else:
                        print(preds)

            except:  # noqa
                print("Inference error...")

    except Exception as e: 
        print(e)
        print("Stopping realsense...")
        rsw.pipeline.stop()

    finally:
        rsw.pipeline.stop()
