import argparse

from datacollection.utils import extract_pose_from_heatmaps
from datacollection.utils import save_heatmaps

from openpose.native.python.utils import str2bool

from rs_py.rs_run_devices import get_rs_parser

from openpose.native.python.args import get_parser as get_op_parser


def get_rs_args():
    args, _ = get_rs_parser().parse_known_args()
    args.rs_steps = 60
    args.rs_fps = 15
    args.rs_image_width = 848
    args.rs_image_height = 480
    # args.rs_color_format = rs.format.bgr8
    # args.rs_depth_format = rs.format.z16
    args.rs_display_frame = 1
    args.rs_save_with_key = False
    args.rs_save_data = True
    args.rs_save_path = 'tmp/data'
    args.rs_use_one_dev_only = False
    args.rs_dev = None
    args.rs_ip = None
    args.rs_verbose = False
    args.rs_autoexposure = True
    args.rs_depth_sensor_autoexposure_limit = 200000.0
    args.rs_enable_ir_emitter = True
    args.rs_ir_emitter_power = 290
    print("========================================")
    print(">>>>> rs_args <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")
    return args


def get_op_args():
    args, _ = get_op_parser().parse_known_args()
    args.op_model_folder = "/usr/local/src/openpose/models/"
    args.op_model_pose = "BODY_25"
    args.op_net_resolution = "-1x368"
    args.op_skel_thres = 0.2
    args.op_max_true_body = 4
    # For skel extraction/tracking in inference.py
    args.op_input_color_image = ""
    args.op_image_width = 848
    args.op_image_height = 480
    # # For 3d skel extraction
    args.op_patch_offset = 5
    # # For 3d skel extraction
    # args.op_extract_3d_skel = False
    # args.op_save_3d_skel = False
    args.op_display = 1.0
    # args.op_display_depth = 0  # Not used
    args.op_rs_delete_image = False
    print("========================================")
    print(">>>>> op_args <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")
    return args


if __name__ == "__main__":

    p = argparse.ArgumentParser(description='Run OPENPOSE on RS images')
    p.add_argument('--extract-pose',
                   type=str2bool,
                   default=False,
                   help='')
    p.add_argument('--save-heatmaps',
                   type=str2bool,
                   default=False,
                   help='')
    args, _ = p.parse_known_args()

    rs_args = get_rs_args()
    op_args = get_op_args()

    if args.extract_pose:
        extract_pose_from_heatmaps(rs_args.rs_save_path, op_args)
    elif args.save_heatmaps:
        save_heatmaps(rs_args.rs_save_path, rs_args, op_args)
    else:
        raise ValueError("No arg given...")
