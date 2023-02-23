import argparse
import cv2
import os
import numpy as np

from typing import Optional

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


def rs_online_inference(args: argparse,
                        pose_extractor: OpenPosePoseExtractor,
                        color_filepath: str,
                        depth_filepath: Optional[str] = None,
                        skeleton_filepath: Optional[str] = None,
                        enable_timer: bool = False) -> bool:
    """Extract pose using openpose per realsense image.

    Args:
        args (argparse): inputs args.
        pose_extractor (OpenPosePoseExtractor): Openpose wrapper class.
        color_filepath (str): path to rgb image.
        depth_filepath (Optional[str], optional): path to depth image.
            If None, the depth data will not be read. Defaults to None.
        skeleton_filepath (Optional[str], optional): path to skeleton csv.
            If None, the extracted pose will not be saved. Defaults to None.
        enable_timer (bool): If true printout timing. Defaults to False.

    Returns:
        bool: False if error, else True
    """

    assert color_filepath is not None

    with Timer(f"data preparation", enable_timer):

        # 1. get the color image
        try:
            image = read_color_file(color_filepath)
        except Exception as e:
            print(e)
            print(f"[WARN] : Error in loading data, skipping {color_filepath}")
            return False

        # 1b. get the depth image
        depth = None
        if args.op_rs_extract_3d_skel and depth_filepath is not None:
            try:
                depth = read_depth_file(depth_filepath)
            except Exception as e:
                print(e)
                print(f"[WARN] : Error in loading data, "
                      f"skipping {depth_filepath}")
                return False

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
            print(f"[WARN] : Error in reshaping image, "
                  f"skipping {color_filepath}")
            return False

    # except Exception as e:
    #     print(e)
    #     print("Stacked data detected")
    #     print(f"Current image shape : {image.shape}")

    #     try:
    #         image = image.reshape(img_h, img_w*3, 3)
    #     except Exception as e:
    #         print(e)
    #         return

    #     data_tuples = []
    #     for i in range(3):
    #         _dir = skel_dir.replace(dev_list[0], dev_list[i])
    #         _path = skel_file.split('.')[0]
    #         _path = _path.split('_')[i]
    #         skel_filepath = os.path.join(_dir, _path + '.csv')
    #         save_path_list = prepare_save_paths(
    #             skel_filepath,
    #             args.op_rs_save_skel,
    #             args.op_rs_save_skel_image,
    #             args.op_rs_save_3d_skel)
    #         j = args.op_rs_image_width * i
    #         k = args.op_rs_image_width * (i+1)
    #         img = image[:, j:k, :]
    #         data_tuples.append([img] + save_path_list)

    with Timer("infer", enable_timer):

        # 3. infer images
        try:
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
                    pose_extractor.predict(image,
                                           kpt_save_path,
                                           skel_image_save_path)

        except Exception as e:
            print(e)
            print(f"[WARN] : Error in pose extraction, "
                  f"skipping {color_filepath}")
            return False


def rs_offline_inference(args: argparse.Namespace):
    """Runs openpose inference on realsense camera in offline mode.

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

    enable_time = False

    PE = OpenPosePoseExtractor(args)

    # 1. If no error
    while not _dict_check(empty_dict):

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

                filepath_dict[dev].append(color_filepaths)

        # 4. loop through devices
        for dev, color_filepaths in filepath_dict:

            # 5. loop through filepaths of color image
            for color_filepath in color_filepaths:

                print(f"[INFO] : {color_filepath}")

                depth_filepath = color_filepath.replace('color', 'depth')
                skel_filepath = skel_filepath.replace('color', 'skeleton')
                skel_filepath = skel_filepath.replace(
                    os.path.splitext(skel_filepath)[1], '.csv')

                # 4. read and infer pose
                status = rs_online_inference(args,
                                             PE,
                                             color_filepath,
                                             depth_filepath,
                                             skel_filepath,
                                             enable_time)

                if args.op_rs_delete_image:
                    os.remove(color_filepath)


def rs_offline_inference_and_tracking(args: argparse.Namespace):
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

    delay_switch = 1
    delay_counter = 0

    display_scale = 0.5
    display_speed = 1

    c = 30

    enable_time = True

    PE = OpenPosePoseExtractor(args)

    if args.op_track_deepsort:
        TK = Tracker('deep_sort', 30//delay_switch)
    elif args.op_track_bytetrack:
        TK = Tracker('byte_tracker', 30//delay_switch)
    elif args.op_track_ocsort:
        TK = Tracker('oc_sort', 30//delay_switch)
    else:
        raise ValueError("Not implemented...")

    # 1. If no error
    while not _dict_check(empty_dict):

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

                # if _c < 300:
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
                status = rs_online_inference(args,
                                             PE,
                                             color_filepath,
                                             depth_filepath,
                                             skel_filepath,
                                             enable_time)

                if TK is not None:
                    if TK.name == 'oc_sort':
                        with Timer("update", enable_time):
                            TK.update(PE.pyop)
                    elif TK.name == 'byte_tracker':
                        with Timer("update", enable_time):
                            TK.update(PE.pyop)
                    else:
                        with Timer("predict", enable_time):
                            TK.predict()
                        with Timer("update", enable_time):
                            TK.update(PE.pyop,
                                      (args.op_rs_image_width,
                                       args.op_rs_image_height))

                PE.display(dev=dev,
                           speed=display_speed,
                           scale=display_scale,
                           image=_image,
                           bounding_box=True,
                           tracks=TK.tracks)

                print(f"Number of tracks : {len(TK.tracks)}")

                # if args.op_rs_delete_image:
                #     os.remove(color_filepath)


if __name__ == "__main__":

    [arg_op, _] = get_parser().parse_known_args()
    # rs_offline_inference(arg_op)
    rs_offline_inference_and_tracking(arg_op)

    print(f"[INFO] : FINISHED")
