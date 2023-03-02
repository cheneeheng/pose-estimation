import argparse
import cv2
import os
import numpy as np
import time

from typing import Optional, Union

from openpose.native.op_py.args import get_parser
from openpose.native.op_py.skeleton import OpenPosePoseExtractor
from openpose.native.op_py.track import Tracker
from openpose.native.op_py.utils import Error
from openpose.native.op_py.utils import Timer
from openpose.native.op_py.utils_rs import get_rs_sensor_dir
from openpose.native.op_py.utils_rs import read_calib_file
from openpose.native.op_py.utils_rs import read_color_file
from openpose.native.op_py.utils_rs import read_depth_file
from openpose.native.op_py.utils_rs import prepare_save_paths


def _dict_check(x: dict):
    c = 0
    for _, v in x.items():
        if v.state:
            c += 1
    return True if c == len(x) else False


def inference(args: argparse.Namespace,
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

    with Timer(f"data preparation", enable_timer):

        # 1. get the color image
        if isinstance(color_src, str):
            try:
                image = read_color_file(color_src)
            except Exception as e:
                print(e)
                print(f"[WARN] : Error loading data, skipping {color_src}")
                return (False, None, None)
        else:
            assert isinstance(color_src, np.ndarray)
            image = color_src

        # 2. reshape images
        try:
            img_h, img_w = args.op_rs_image_height, args.op_rs_image_width
            image = image.reshape(img_h, img_w, 3)
            save_path_list = prepare_save_paths(skeleton_filepath,
                                                args.op_rs_save_skel,
                                                args.op_rs_save_skel_image)
        except Exception as e:
            print(e)
            print(f"[WARN] : Error in reshaping image...")
            if isinstance(color_src, str):
                print(f"skipping {color_src}")
            return (False, None, None)

    with Timer("infer", enable_timer):

        # 3. infer images
        try:
            pose_extractor.predict(image, save_path_list[0], save_path_list[2])
            return (True, pose_extractor.pyop.pose_keypoints,
                    pose_extractor.pyop.pose_scores)
        except Exception as e:
            print(e)
            print(f"[WARN] : Error in pose extraction...")
            if isinstance(color_src, str):
                print(f"skipping {color_src}")
            return (False, None, None)


def rs_inference(args: argparse.Namespace,
                 pose_extractor: OpenPosePoseExtractor,
                 color_src: Union[np.ndarray, str],
                 depth_src: Optional[Union[np.ndarray, str]] = None,
                 skeleton_filepath: Optional[str] = None,
                 enable_timer: bool = False,
                 profile: bool = False) -> bool:
    """Extract pose using openpose per realsense image.

    Args:
        args (argparse.Namespace): inputs args.
        pose_extractor (OpenPosePoseExtractor): Openpose wrapper class.
        color_src (Union[np.ndarray, str]): rgb image or path to rgb image.
        depth_src (Optional[Union[np.ndarray, str]], optional): depth image or
            path to depth image. If None, the depth data will not be used.
            Defaults to None.
        skeleton_filepath (Optional[str], optional): path to skeleton csv.
            If None, the extracted pose will not be saved. Defaults to None.
        enable_timer (bool): If true printout timing. Defaults to False.

    Returns:
        bool: False if error, else True
    """

    assert color_src is not None

    with Timer(f"data preparation", enable_timer):

        # 1. get the color image
        if isinstance(color_src, str):
            try:
                image = read_color_file(color_src)
            except Exception as e:
                print(e)
                print(f"[WARN] : Error loading data, skipping {color_src}")
                return False, None
        else:
            assert isinstance(color_src, np.ndarray)
            image = color_src

        # 1b. get the depth image
        depth = None
        if args.op_rs_extract_3d_skel and depth_src is not None:
            if isinstance(depth_src, str):
                try:
                    depth = read_depth_file(depth_src)
                except Exception as e:
                    print(e)
                    print(f"[WARN] : Error loading data, skipping {depth_src}")
                    return False, None
            else:
                assert isinstance(depth_src, np.ndarray)
                depth = depth_src

        # 2. reshape images
        try:
            img_h, img_w = args.op_rs_image_height, args.op_rs_image_width
            image = image.reshape(img_h, img_w, 3)
            save_path_list = prepare_save_paths(skeleton_filepath,
                                                args.op_rs_save_skel,
                                                args.op_rs_save_skel_image,
                                                args.op_rs_save_3d_skel)
            data_tuples = [[image] + save_path_list]

        except Exception as e:
            print(e)
            print(f"[WARN] : Error in reshaping image...")
            if isinstance(color_src, str):
                print(f"skipping {color_src}")
            return False, None

    with Timer("infer", enable_timer):

        # 3. infer images
        try:
            prof = 0
            for (image, kpt_save_path, kpt_3d_save_path, skel_image_save_path
                 ) in data_tuples:
                if args.op_rs_extract_3d_skel:
                    main_dir = os.path.dirname(os.path.dirname(kpt_save_path))
                    intr_mat = read_calib_file(main_dir + "/calib/calib.csv")
                    pose_extractor.predict_3d(image,
                                              depth,
                                              intr_mat,
                                              kpt_save_path,
                                              kpt_3d_save_path,
                                              skel_image_save_path)
                else:
                    if profile:
                        s = time.time()
                    pose_extractor.predict(image,
                                           kpt_save_path,
                                           skel_image_save_path)
                    if profile:
                        prof += time.time()-s
            return True, prof

        except Exception as e:
            print(e)
            print(f"[WARN] : Error in pose extraction...")
            if isinstance(color_src, str):
                print(f"skipping {color_src}")
            return False, None


def rs_offline_inference(args: argparse.Namespace, profile: bool = False):
    """Runs openpose inference and tracking on realsense camera in offline mode.

    Reads realsense image files under the `base_path` arg and extracts pose
    from the images using openpose.

    Args:
        args (argparse.Namespace): CLI arguments
    """

    assert os.path.isdir(args.op_rs_dir), f'{args.op_rs_dir} does not exist...'

    base_path = args.op_rs_dir
    dev_trial_color_dir = get_rs_sensor_dir(base_path, 'color')
    dev_list = list(dev_trial_color_dir.keys())

    empty_dict = {i: Error() for i in dev_list}
    end_loop = False

    delay_switch = 1
    delay_counter = 0

    display_scale = 0.5
    display_speed = 1

    runtime = {'PE': [], 'TK': []}

    c = 30

    enable_time = True

    PE = OpenPosePoseExtractor(args)

    if args.op_track_deepsort:
        TK = Tracker('deep_sort', 30//delay_switch)
    elif args.op_track_bytetrack:
        TK = Tracker('byte_tracker', 30//delay_switch)
    elif args.op_track_ocsort:
        TK = Tracker('oc_sort', 30//delay_switch)
    elif args.op_track_strongsort:
        TK = Tracker('strong_sort', 30//delay_switch)
    else:
        TK = None

    # 1. If no error
    while not _dict_check(empty_dict) and not end_loop:

        filepath_dict = {i: [] for i in dev_list}

        # 2. loop through devices
        for dev, trial_color_dir in dev_trial_color_dir.items():

            # 3. loop through trials
            for trial, color_dir in trial_color_dir.items():

                color_filepaths = [os.path.join(color_dir, i)
                                   for i in sorted(os.listdir(color_dir))]

                if len(color_filepaths) == 0:
                    print(f"[INFO] : {color_dir} is empty...")
                    empty_dict[dev].counter += 1
                    if empty_dict[dev].counter > 300:
                        print("[INFO] Retried 300 times and no new files...")
                        empty_dict[dev].state = True
                    continue

                filepath_dict[dev] += color_filepaths

        # 4. loop through devices
        for dev, color_filepaths in filepath_dict.items():

            _c = 0

            # 5. loop through filepaths of color image
            for color_filepath in color_filepaths:

                # if _c < 480:
                #     _c += 1
                #     continue

                print(f"[INFO] : {color_filepath}")

                depth_filepath = color_filepath.replace('color', 'depth')
                skel_filepath = color_filepath.replace('color', 'skeleton')
                skel_filepath = skel_filepath.replace(
                    os.path.splitext(skel_filepath)[1], '.csv')

                _image = read_color_file(color_filepath)

                if TK is not None:
                    if delay_counter > 1:
                        delay_counter -= 1
                        TK.no_measurement_predict_and_update()
                        PE.display(dev=dev,
                                   speed=display_speed,
                                   scale=display_scale,
                                   image=_image,
                                   bounding_box=True,
                                   tracks=TK.tracker.tracks)
                        continue
                    else:
                        delay_counter = delay_switch

                # 4. read and infer pose
                status = rs_inference(args,
                                      PE,
                                      color_filepath,
                                      depth_filepath,
                                      skel_filepath,
                                      enable_time,
                                      profile)
                runtime['PE'].append(1/status[1])

                s = time.time()
                if TK is not None:
                    if TK.name == 'deep_sort' or TK.name == 'strong_sort':
                        with Timer("update", enable_time):
                            TK.update(PE.pyop,
                                      (args.op_rs_image_width,
                                       args.op_rs_image_height))
                    elif TK.name == 'byte_tracker' or TK.name == 'oc_sort':
                        with Timer("update", enable_time):
                            TK.update(PE.pyop)
                runtime['TK'].append(1/(time.time()-s))

                # status = PE.display(dev=dev,
                #                     speed=display_speed,
                #                     scale=display_scale,
                #                     image=_image,
                #                     bounding_box=False,
                #                     tracks=TK.tracks)

                img = PE.pyop.opencv_image
                img = PE.pyop._draw_skeleton_bounding_box_image(img)
                img = PE.pyop._draw_bounding_box_tracking_image(img, TK.tracks)
                img = cv2.resize(img, (int(img.shape[1]*display_scale),
                                       int(img.shape[0]*display_scale)))
                _img = cv2.resize(_image, (int(_image.shape[1]*display_scale),
                                           int(_image.shape[0]*display_scale)))
                img = np.concatenate([img, _img], axis=0)
                save_file = color_filepath.replace('color',
                                                   f'color_{TK.name}')
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
                if status[0] and save_file is not None:
                    cv2.imwrite(save_file, img)

                print(f"Number of tracks : {len(TK.tracks)}")
                if profile:
                    print(f"FPS PE : {sum(runtime['PE'])/len(runtime['PE'])}")
                    print(f"FPS TK : {sum(runtime['TK'])/len(runtime['TK'])}")

                # if args.op_rs_delete_image:
                #     os.remove(color_filepath)

            end_loop = True


if __name__ == "__main__":

    [arg_op, _] = get_parser().parse_known_args()
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_deepsort")
    arg_op.op_track_deepsort = True
    rs_offline_inference(arg_op, profile=True)
    arg_op.op_track_deepsort = False
    [arg_op, _] = get_parser().parse_known_args()
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_bytetrack")
    arg_op.op_track_bytetrack = True
    rs_offline_inference(arg_op, profile=True)
    arg_op.op_track_bytetrack = False
    [arg_op, _] = get_parser().parse_known_args()
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_ocsort")
    arg_op.op_track_ocsort = True
    rs_offline_inference(arg_op, profile=True)
    arg_op.op_track_ocsort = False
    [arg_op, _] = get_parser().parse_known_args()
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_strongsort")
    arg_op.op_track_strongsort = True
    rs_offline_inference(arg_op, profile=True)
    arg_op.op_track_strongsort = False
    print(f"[INFO] : FINISHED")
