import argparse
import cv2
import numpy as np
import os
import time
from tqdm import trange

from openpose.native import PyOpenPoseNative


def str2bool(v) -> bool:
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Extract 3D skeleton using OPENPOSE')
    # MAIN
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
                        type=str2bool,
                        default=False,
                        help='whether to use coordinate system of NTU')
    parser.add_argument('--op-save-skel',
                        type=str2bool,
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
    # RUN OPTIONS
    parser.add_argument('--op-test-runtime',
                        type=str2bool,
                        default=False,
                        help='if true, test runtime of openpose.')
    parser.add_argument('--op-rs-offline-inference',
                        type=str2bool,
                        default=False,
                        help='if true, run openpose on saved images from rs.')
    parser.add_argument('--op-rs-delete-image',
                        type=str2bool,
                        default=False,
                        help='if true, deletes the rs image used in inference.')
    parser.add_argument('--op-rs-dir',
                        type=str,
                        default='-1',
                        help='path to folder with saved rs data.')
    parser.add_argument('--op-rs-image-width',
                        type=int,
                        default=848,
                        help='image width in px')
    parser.add_argument('--op-rs-image-height',
                        type=int,
                        default=480,
                        help='image height in px')
    return parser


def test_op_runtime(args: argparse.Namespace):
    image_path = "openpose/pexels-photo-4384679.jpeg"
    target_path = "openpose/output/inference_native"
    os.makedirs(target_path, exist_ok=True)
    params = dict(
        model_folder=args.op_model_folder,
        model_pose=args.op_model_pose,
        net_resolution=args.op_net_resolution,
    )
    pyop = PyOpenPoseNative(params,
                            args.op_skel_thres,
                            args.op_max_true_body,
                            args.op_patch_offset,
                            args.op_ntu_format)
    pyop.initialize()
    t_total = 0
    N = 1000
    image = cv2.imread(image_path)
    image = cv2.resize(image, (384, 384))
    for _ in trange(N):
        t_start = time.time()
        pyop.predict(image)
        pyop.filter_prediction()
        # pyop.display(1, 'dummy')
        pyop.save_pose_keypoints(f'{target_path}/predictions.txt')
        t_total += time.time() - t_start
    print(f"Average inference time over {N} trials : {t_total/N}s")


# Based on:
# https://github.com/cheneeheng/realsense-simple-wrapper/blob/main/rs_py/rs_view_raw_data.py
def get_rs_sensor_dir(base_path: str, sensor: str) -> dict:
    path = {}
    for device in sorted(os.listdir(base_path)):
        path[device] = {}
        device_path = os.path.join(base_path, device)
        for ts in sorted(os.listdir(device_path)):
            path[device][ts] = os.path.join(base_path, device, ts, sensor)
    return path


# From:
# https://github.com/cheneeheng/realsense-simple-wrapper/blob/main/rs_py/rs_view_raw_data.py
def _get_brg_from_yuv(data_array: np.ndarray) -> np.ndarray:
    # Input: Intel handle to 16-bit YU/YV data
    # Output: BGR8
    UV = np.uint8(data_array >> 8)
    Y = np.uint8((data_array << 8) >> 8)
    YUV = np.zeros((data_array.shape[0], 2), 'uint8')
    YUV[:, 0] = Y
    YUV[:, 1] = UV
    YUV = YUV.reshape(-1, 1, 2)
    BGR = cv2.cvtColor(YUV, cv2.COLOR_YUV2BGR_YUYV)
    BGR = BGR.reshape(-1, 3)
    return BGR


# From:
# https://github.com/cheneeheng/realsense-simple-wrapper/blob/main/rs_py/rs_view_raw_data.py
def read_color_file(color_file: str) -> np.ndarray:
    if color_file.endswith('.bin'):
        with open(color_file, 'rb') as f:
            image = np.fromfile(f, np.uint8)
    else:
        image = np.load(color_file)
        if image.dtype == np.dtype(np.uint8):
            pass
        elif image.dtype == np.dtype(np.uint16):
            image = _get_brg_from_yuv(image.reshape(-1))
        else:
            raise ValueError("Unknown data type :", image.dtype)
    return image


def rs_offline_inference(args: argparse.Namespace):
    """Runs openpose offline by looking at images found in the image_path arg.

    Args:
        args (argparse.Namespace): CLI arguments
    """
    def _create_save_path(color_dir: str, color_file: str):
        save_dir = color_dir.replace('color', 'skeleton')
        save_file = color_file.replace('npy', 'csv')
        save_path = os.path.join(save_dir, save_file)
        return save_dir, save_path

    assert args.op_rs_offline_inference, f'op_rs_offline_inference is False...'
    assert os.path.isdir(args.op_rs_dir), f'{args.op_rs_dir} does not exist...'

    params = dict(
        model_folder=args.op_model_folder,
        model_pose=args.op_model_pose,
        net_resolution=args.op_net_resolution,
    )
    pyop = PyOpenPoseNative(params,
                            args.op_skel_thres,
                            args.op_max_true_body,
                            args.op_patch_offset,
                            args.op_ntu_format)
    pyop.initialize()

    base_path = args.op_rs_dir
    dev_trial_color_dir = get_rs_sensor_dir(base_path, 'color')
    dev_list = list(dev_trial_color_dir.keys())

    while True:

        c = 0
        for dev, trial_color_dir in dev_trial_color_dir.items():

            for trial, color_dir in trial_color_dir.items():

                color_files = sorted(os.listdir(color_dir))

                if len(color_files) == 0:
                    print(f"[INFO] : {color_dir} is empty...")
                    c += 1
                    continue

                # get the file that has not been inferred on.
                for i, color_file in enumerate(color_files):
                    color_filepath = os.path.join(color_dir, color_file)
                    save_dir, save_path = _create_save_path(color_dir,
                                                            color_file)
                    if not os.path.exists(save_path):
                        i = i-1
                        break

                if i+1 == len(color_files):
                    print(f"[INFO] : {color_dir} is fully evaluated...")
                    c += 1
                    continue

                print(f"[INFO] : {len(color_files)-i} files left...")

                try:
                    image = read_color_file(color_filepath)

                except Exception as e:
                    print(e)
                    print("[WARN] : Error in loading data, will retry....")
                    continue

                try:
                    image = image.reshape(args.op_rs_image_height,
                                          args.op_rs_image_width, 3)
                    pyop.predict(image)
                    pyop.filter_prediction()
                    os.makedirs(save_dir, exist_ok=True)
                    pyop.save_pose_keypoints(save_path)
                    print(f"[INFO] : OP output saved in {save_path}")

                except Exception as e:
                    print(e)
                    print("Stacked data detected")

                    try:
                        image = image.reshape(args.op_rs_image_height,
                                              args.op_rs_image_width*3, 3)
                        for i in range(3):
                            j = args.op_rs_image_width * i
                            k = args.op_rs_image_width * (i+1)
                            image = image[:, j:k, :]
                            pyop.predict(image)
                            pyop.filter_prediction()
                            _dir = save_dir.replace(dev_list[0], dev_list[i])
                            os.makedirs(_dir, exist_ok=True)
                            _path = save_path.split('/')[-1]
                            _path = _path.split('.')[0]
                            _path = _path.split('_')[i]
                            _path = os.path.join(_dir, _path + '.csv')
                            pyop.save_pose_keypoints(_path)
                            print(f"[INFO] : OP output saved in {_path}")

                    except Exception as e:
                        print(e)
                        continue

                if args.op_rs_delete_image:
                    os.remove(color_filepath)


if __name__ == "__main__":
    [arg_op, _] = get_parser().parse_known_args()
    if arg_op.op_test_runtime:
        test_op_runtime(arg_op)
    elif arg_op.op_rs_offline_inference:
        rs_offline_inference(arg_op)

    print(f"[INFO] : FINISHED")
