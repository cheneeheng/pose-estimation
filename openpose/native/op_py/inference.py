import argparse
import numpy as np
from typing import Optional, Union

from openpose.native.op_py.args import get_parser
from openpose.native.op_py.skeleton import OpenPosePoseExtractor
from openpose.native.op_py.utils import Timer
from openpose.native.op_py.utils_rs import read_color_file
from openpose.native.op_py.utils_rs import prepare_save_paths


def extract_2dskeletons(args: argparse.Namespace,
                        pose_extractor: OpenPosePoseExtractor,
                        color_src: Union[np.ndarray, str],
                        skeleton_filepath: Optional[str] = None,
                        enable_timer: bool = False) -> bool:
    """Extract pose using openpose per realsense image.

    Args:
        args (argparse.Namespace): inputs args.
        pose_extractor (OpenPosePoseExtractor): Openpose wrapper class.
        color_src (Union[np.ndarray, str]): rgb image or path to rgb image.
        skeleton_filepath (Optional[str], optional): path to skeleton csv.
            If None, the extracted pose will not be saved. Defaults to None.
        enable_timer (bool): If true printout timing. Defaults to False.

    Returns:
        bool: False if error, else True
    """
    assert color_src is not None

    with Timer(f"data preparation", enable_timer, printout=enable_timer):

        # 1. get the color image
        if isinstance(color_src, str):
            try:
                image = read_color_file(color_src)
            except Exception as e:
                print(e)
                print(f"[WARN] : Error loading data, skipping {color_src}")
                return {'status': False, 'keypoints': None, 'scores': None}
        else:
            assert isinstance(color_src, np.ndarray)
            image = color_src

        # 2. reshape images
        try:
            img_h, img_w = args.op_rs_image_height, args.op_rs_image_width
            image = image.reshape(img_h, img_w, 3)
            save_path_list = prepare_save_paths(skeleton_filepath,
                                                args.op_save_skel,
                                                args.op_save_skel_image)
        except Exception as e:
            print(e)
            print(f"[WARN] : Error in reshaping image...")
            if isinstance(color_src, str):
                print(f"skipping {color_src}")
            return {'status': False, 'keypoints': None, 'scores': None}

    with Timer("infer", enable_timer, printout=enable_timer):

        # 3. infer images
        try:
            pose_extractor.predict(image, save_path_list[0], save_path_list[2])

            return {'status': True,
                    'keypoints': pose_extractor.pyop.pose_keypoints,
                    'scores': pose_extractor.pyop.pose_scores}
        except Exception as e:
            print(e)
            print(f"[WARN] : Error in pose extraction...")
            if isinstance(color_src, str):
                print(f"skipping {color_src}")
            return {'status': False, 'keypoints': None, 'scores': None}


if __name__ == "__main__":

    [arg_op, _] = get_parser().parse_known_args()
    PE = OpenPosePoseExtractor(arg_op)
    enable_timer = True
    extract_2dskeletons(arg_op,
                        PE,
                        arg_op.op_color_image,
                        arg_op.op_skel_file,
                        enable_timer)

    print(f"[INFO] : FINISHED")
