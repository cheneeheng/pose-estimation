import argparse
import numpy as np
import os

from datetime import datetime

from openpose.native import PyOpenPoseNative
from realsense import StoragePaths
from realsense import get_rs_parser
from realsense import initialize_rs_devices


class OpenposeStoragePaths(StoragePaths):
    def __init__(self, device_sn: str = ''):
        super().__init__()
        base_path = '/data/openpose'
        date_time = datetime.now().strftime("%y%m%d%H%M%S")
        self.calib = f'{base_path}/calib/{date_time}_dev{device_sn}'
        self.color = f'{base_path}/color/{date_time}_dev{device_sn}'
        self.depth = f'{base_path}/depth/{date_time}_dev{device_sn}'
        self.skeleton = f'{base_path}/skeleton/{date_time}_dev{device_sn}'
        self.timestamp = f'{base_path}/timestamp/{date_time}_dev{device_sn}'
        self.timestamp_file = os.path.join(self.timestamp, 'timestamp.txt')
        os.makedirs(self.calib, exist_ok=True)
        os.makedirs(self.color, exist_ok=True)
        os.makedirs(self.depth, exist_ok=True)
        os.makedirs(self.skeleton, exist_ok=True)
        os.makedirs(self.timestamp, exist_ok=True)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Extract 3D skeleton using OPENPOSE')
    parser.add_argument('--op-model-folder',
                        type=str,
                        default="/usr/local/src/openpose/models/",
                        help='folder with trained openpose models.')
    parser.add_argument('--op-model-pose',
                        type=str,
                        default="BODY_25",
                        help='pose model name')
    parser.add_argument('--op-net-resolution',
                        type=str,
                        default="-1x368",
                        help='resolution of input to openpose.')
    parser.add_argument('--op-skel-thres',
                        type=float,
                        default=0.5,
                        help='threshold for valid skeleton.')
    parser.add_argument('--op-max-true-body',
                        type=int,
                        default=2,
                        help='max number of skeletons to save.')
    parser.add_argument('--op-patch-offset',
                        type=int,
                        default=2,
                        help='offset of patch used to determine depth')
    parser.add_argument('--op-ntu-format',
                        type=bool,
                        default=False,
                        help='whether to use coordinate system of NTU')
    parser.add_argument('--op-save-skel',
                        default=True,
                        help='if true, save 3d skeletons.')
    parser.add_argument('--op-display',
                        type=int,
                        default=0,
                        help='scale for displaying skel images.')
    parser.add_argument('--op-display-depth',
                        type=int,
                        default=0,
                        help='scale for displaying skel images with depth.')
    parser.add_argument('--op-debug',
                        type=int,
                        default=0,
                        help='0: no debug, '
                             '1: no openpose')
    return parser


if __name__ == "__main__":

    [arg_op, _] = get_parser().parse_known_args()
    [arg_rs, _] = get_rs_parser().parse_known_args()

    # 0. Initialize ------------------------------------------------------------
    # OPENPOSE
    if arg_op.op_debug != 1:
        params = dict(
            model_folder=arg_op.op_model_folder,
            model_pose=arg_op.op_model_pose,
            net_resolution=arg_op.op_net_resolution,
        )
        pyop = PyOpenPoseNative(params,
                                arg_op.op_skel_thres,
                                arg_op.op_max_true_body,
                                arg_op.op_patch_offset,
                                arg_op.op_ntu_format)
        pyop.initialize()

    # REALSENSE
    rsw = initialize_rs_devices(arg_rs, OpenposeStoragePaths)

    state = True
    empty_skeleton_3d = np.zeros((25, 3))
    print("Starting frame capture loop...")
    try:
        while state:

            # 1. Get rs data ---------------------------------------------------
            print("Running...")
            frames = rsw.run(display=arg_rs.display_frame)
            if not len(frames) > 0:
                continue

            for dev_sn, data_dict in frames.items():
                color_image = frames[dev_sn]['color']
                depth_image = frames[dev_sn]['depth']
                timestamp = frames[dev_sn]['timestamp']
                calib = frames[dev_sn]['calib']

                # 2. Predict pose ----------------------------------------------
                # bgr format
                if arg_op.op_debug != 1:
                    pyop.predict(color_image)

                    # 3. Save data ---------------------------------------------
                    if arg_op.op_save_skel:
                        intr_mat = calib['color'][0]['intrinsic_mat']
                        skel_save_path = os.path.join(
                            rsw.storage_paths[dev_sn].skeleton,
                            f'{timestamp:020d}' + '.txt'
                        )
                        pyop.convert_to_3d(
                            intr_mat=intr_mat,
                            depth_image=depth_image,
                            empty_pose_keypoints_3d=empty_skeleton_3d,
                            save_path=skel_save_path
                        )

                    if arg_op.op_display_depth > 0:
                        stop = pyop.display(arg_op.op_display_depth,
                                            dev_sn, depth_image)
                        if stop:
                            break
                    elif arg_op.op_display > 0:
                        stop = pyop.display(arg_op.op_display,
                                            dev_sn, depth_image)
                        if stop:
                            break

    except Exception as e:
        print(e)
        print("Stopping realsense...")
        rsw.stop()

    finally:
        rsw.stop()
