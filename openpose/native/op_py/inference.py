import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm
from typing import Optional, Union

from openpose.native.op_py.args import get_parser
from openpose.native.op_py.skeleton import OpenPosePoseExtractor
from openpose.native.op_py.track import Tracker
from openpose.native.op_py.utils import Timer
from openpose.native.op_py.utils_rs import read_color_file
from openpose.native.op_py.utils_rs import read_depth_file
from openpose.native.op_py.utils_rs import prepare_save_paths


def extract_2dskeletons(args: argparse.Namespace,
                        pose_extractor: OpenPosePoseExtractor,
                        color_src: Union[np.ndarray, str],
                        depth_src: Optional[Union[np.ndarray, str]] = None,
                        intr_mat: Optional[Union[np.ndarray, str]] = None,
                        skeleton_filepath: Optional[str] = None,
                        enable_timer: bool = False) -> dict:
    """Extract pose using openpose on an image.

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
        float: data_prep_time, -1 if error
        float: infer_time, -1 if error
    """
    assert color_src is not None

    with Timer(f"data preparation", enable_timer, printout=False) as t1:

        # 1a. get the color image
        if isinstance(color_src, str):
            try:
                image = read_color_file(color_src)
            except Exception as e:
                print(e)
                print(f"[WARN] : Error loading data, skipping {color_src}")
                return {'status': False,
                        'data_prep_time': -1,
                        'infer_time': -1}
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
                    return {'status': False,
                            'data_prep_time': -1,
                            'infer_time': -1}
            else:
                assert isinstance(depth_src, np.ndarray)
                depth = depth_src

        # 2. reshape images
        try:
            img_h, img_w = args.op_rs_image_height, args.op_rs_image_width
            image = image.reshape(img_h, img_w, 3)
            save_path_list = prepare_save_paths(skeleton_filepath,
                                                args.op_save_skel,
                                                args.op_save_skel_image,
                                                args.op_rs_save_3d_skel)
        except Exception as e:
            print(e)
            print(f"[WARN] : Error in reshaping image...")
            if isinstance(color_src, str):
                print(f"skipping {color_src}")
            return {'status': False,
                    'data_prep_time': -1,
                    'infer_time': -1}

    with Timer("infer", enable_timer, printout=False) as t2:

        # 3. infer images
        try:
            if args.op_rs_extract_3d_skel:
                # main_dir = os.path.dirname(os.path.dirname(kpt_save_path))
                # intr_mat = read_calib_file(main_dir + "/calib/calib.csv")
                pose_extractor.predict_3d(image,
                                          depth,
                                          intr_mat,
                                          save_path_list[0],
                                          save_path_list[1],
                                          save_path_list[2])
            else:
                pose_extractor.predict(image,
                                       depth,
                                       save_path_list[0],
                                       save_path_list[2])
        except Exception as e:
            print(e)
            print(f"[WARN] : Error in pose extraction...")
            if isinstance(color_src, str):
                print(f"skipping {color_src}")
            return {'status': False,
                    'data_prep_time': -1,
                    'infer_time': -1}

    return {'status': True,
            'data_prep_time': t1.duration,
            'infer_time': t2.duration}


def extract_2dskeletons_and_track_offline(args: argparse.Namespace):
    """Runs openpose inference and tracking in offline mode.

    Reads image files under the `color_image` arg and extracts pose
    from the images using openpose.

    Args:
        args (argparse.Namespace): CLI arguments
    """

    assert os.path.isdir(args.op_color_image), \
        f'{args.op_color_image} does not exist...'

    base_path = args.op_color_image
    filepaths = sorted([os.path.join(base_path, f"{i:0>20}")
                        for i in os.listdir(base_path)])

    delay_switch = 1
    delay_counter = 0

    display_speed = 1

    runtime = {'PE': [], 'TK': []}

    enable_timer = True

    PE = OpenPosePoseExtractor(args)

    if args.op_track_deepsort:
        TK = Tracker('deep_sort', args, 30//delay_switch)
    elif args.op_track_bytetrack:
        TK = Tracker('byte_tracker', args,  30//delay_switch)
    elif args.op_track_ocsort:
        TK = Tracker('oc_sort', args,  30//delay_switch)
    elif args.op_track_strongsort:
        TK = Tracker('strong_sort', args,  30//delay_switch)
    else:
        TK = None

    # 1. loop through devices
    _c = 0
    tqdm_bar = tqdm(filepaths, dynamic_ncols=True)

    # 2. loop through filepaths of color image
    for idx, color_filepath in enumerate(tqdm_bar):

        if idx == 15:
            runtime = {'PE': [], 'TK': []}

        # if _c < 480:
        #     _c += 1
        #     continue

        color_file = color_filepath.split('/')[-1]
        color_filepath = color_filepath.replace(color_file,
                                                color_file.lstrip('0'))

        paths = color_filepath.split('/')
        skel_filepath = color_filepath.replace(paths[-2], 'skeleton')
        skel_filepath = skel_filepath.replace(
            os.path.splitext(skel_filepath)[1], '.csv')

        if TK is not None:
            if delay_counter > 1:
                delay_counter -= 1
                _image = read_color_file(color_filepath)
                TK.no_measurement_predict_and_update()
                PE.display(dev='000',
                           speed=display_speed,
                           scale=args.op_display,
                           image=_image,
                           bounding_box=True,
                           tracks=TK.tracker.tracks)
                continue
            else:
                delay_counter = delay_switch

        # 3. read and infer pose
        infer_out = extract_2dskeletons(args=args,
                                        pose_extractor=PE,
                                        color_src=color_filepath,
                                        skeleton_filepath=skel_filepath,
                                        enable_timer=enable_timer)
        if PE.pyop.pose_bounding_box is None:
            boxes = None
            scores = None
            keypoints = None
            heatmaps = None
        else:
            boxes = PE.pyop.pose_bounding_box.copy()
            scores = PE.pyop.pose_scores.copy()
            keypoints = PE.pyop.pose_keypoints.copy()
            heatmaps = PE.pyop.pose_heatmaps.copy()
        runtime['PE'].append(1/(infer_out['infer_time']+1e-8))

        with Timer("update", enable_timer) as t:
            if boxes is not None and TK is not None:
                TK.update(boxes, scores, keypoints, heatmaps,
                          (args.op_rs_image_width,
                           args.op_rs_image_height))
        runtime['TK'].append(1/(t.duration+1e-8))

        status = PE.display(win_name='000',
                            speed=display_speed,
                            scale=args.op_display,
                            image=None,
                            bounding_box=False,
                            tracks=TK.tracks)

        if args.op_save_result_image and boxes is not None:
            img = PE.pyop.opencv_image
            if TK is not None:
                img = PE.pyop._draw_skeleton_text_image(img, scores)
                img = PE.pyop._draw_bounding_box_tracking_image(img, TK.tracks)  # noqa
            else:
                img = PE.pyop._draw_skeleton_text_image(img, scores)
                img = PE.pyop._draw_skeleton_bounding_box_image(img, boxes)
            img = cv2.resize(img,
                             (int(img.shape[1]*args.op_display),
                              int(img.shape[0]*args.op_display)))
            paths = color_filepath.split('/')
            save_file = color_filepath.replace(
                paths[-2], f'{paths[-2]}_{TK.name}')
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            cv2.imwrite(save_file, img)

        tqdm_bar.set_description(
            f"Image : {color_filepath.split('/')[-1]} | "
            f"#Skel filtered : {PE.pyop.filtered_skel} | "
            f"#Tracks : {len(TK.tracks)} | "
            f"Prep time : {infer_out['data_prep_time']:.3f} | "
            f"Pose time : {infer_out['infer_time']:.3f} | "  # noqa
            f"Track time : {t.duration:.3f} | "
            f"A-FPS PE : {sum(runtime['PE'])/len(runtime['PE']):.3f} | "  # noqa
            f"A-FPS TK : {sum(runtime['TK'])/len(runtime['TK']):.3f}"
        )


if __name__ == "__main__":

    arg_op, _ = get_parser().parse_known_args()

    if os.path.isdir(arg_op.op_color_image):
        extract_2dskeletons_and_track_offline(arg_op)
    else:
        PE = OpenPosePoseExtractor(arg_op)
        enable_timer = True
        extract_2dskeletons(args=arg_op,
                            pose_extractor=PE,
                            color_src=arg_op.op_color_image,
                            skeleton_filepath=arg_op.op_skel_file,
                            enable_timer=enable_timer)

    print(f"[INFO] : FINISHED")
