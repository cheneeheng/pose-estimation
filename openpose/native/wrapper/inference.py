import argparse
import cv2
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
    parser.add_argument('--op-test-runtime',
                        type=str2bool,
                        default=True,
                        help='if true, test runtime of openpose.')
    return parser


def save_prediction(prediction: list, save_path: str) -> None:
    for skel in prediction:
        save_str = ",".join([str(i) for pos in skel for i in pos])
        with open(save_path, 'a+') as f:
            f.write(f'{save_str}\n')


def test_op_runtime(arg_op: argparse.Namespace):
    image_path = "openpose/pexels-photo-4384679.jpeg"
    target_path = "openpose/output/inference_native"
    os.makedirs(target_path, exist_ok=True)
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
    t_total = 0
    N = 1000
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    for _ in trange(N):
        t_start = time.time()
        pyop.predict(image)
        # pyop.display(1, 'dummy')
        # pred = pyop.pose_keypoints_filtered
        # save_prediction(pred, f'{target_path}/predictions.txt')
        t_total += time.time() - t_start
    print(f"Average inference time over {N} trials : {t_total/N}s")


if __name__ == "__main__":
    [arg_op, _] = get_parser().parse_known_args()
    if arg_op.op_test_runtime:
        test_op_runtime(arg_op)
