import argparse
import cv2
import numpy as np
import os

from openpose.native import OpenposeStoragePaths
from openpose.native import display_skel
from openpose.native import save_skel
from openpose.native import PyOpenPoseNative
from realsense import RealsenseWrapper

from infer.inference import parse_arg
from infer.inference import get_parser as get_agcn_parser
from infer.inference import init_file_and_folders
from infer.inference import init_preprocessor
from infer.inference import init_model
from infer.inference import model_inference
from infer.inference import filter_logits


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

    # REALSENSE +  STORAGE
    rsw = RealsenseWrapper()
    rsw.stream_config.fps = 30
    rsw.available_devices = rsw.available_devices[0:1]
    rsw.initialize()
    rsw.set_storage_paths(OpenposeStoragePaths)
    rsw.save_calibration()

    dev_sn = rsw.available_devices[0]

    state = True
    empty_skeleton_3d = np.zeros((25, 3))

    mva = 5
    mva_list = []

    try:
        while state:

            # 1. Get rs data ---------------------------------------------------
            # state, color_image, depth_image, timestamp = rsw.run(
            frames = rsw.run(display=op_arg.display_rs)
            state = len(frames) > 0
            if not state:
                continue

            color_image = frames[dev_sn]['color']
            depth_image = frames[dev_sn]['depth']
            timestamp = frames[dev_sn]['timestamp']
            calib = frames[dev_sn]['calib']

            # 2. Predict pose --------------------------------------------------
            # bgr format
            pyop.predict(color_image)

            # 3. Save data -------------------------------------------------
            if op_arg.save_skel:
                intr_mat = calib['color'][0]['intrinsic_mat']
                skel_save_path = os.path.join(
                    rsw.storage_paths[dev_sn].skeleton,
                    f'{timestamp:020d}' + '.txt'
                )
                save_skel(pyop, op_arg, depth_image, intr_mat,
                          empty_skeleton_3d, skel_save_path)

            if op_arg.display_skel:
                # state = display_skel(pyop, dev_sn)
                keypoint_image = pyop.opencv_image
                keypoint_image = cv2.flip(keypoint_image, 1)
                cv2.putText(keypoint_image,
                            "KP (%) : " + str(round(max(pyop.pose_scores), 2)),
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
                cv2.moveWindow("depth_keypoint_overlay", 1500, 300)
                key = cv2.waitKey(30)

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
        rsw.stop()

    finally:
        rsw.stop()
