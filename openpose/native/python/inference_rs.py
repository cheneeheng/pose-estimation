import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm
from threading import Thread
from queue import Queue
import multiprocessing as mp
from typing import Optional, Union

from openpose.native.python.args import get_parser
from openpose.native.python.inference import extract_2dskeletons
from openpose.native.python.skeleton import PyOpenPoseNative as PYOP
from openpose.native.python.skeleton import OpenPosePoseExtractor
from openpose.native.python.utils import dict_check
from openpose.native.python.utils import Error
from openpose.native.python.utils import Timer
from openpose.native.python.utils_rs import get_rs_sensor_dir
from openpose.native.python.utils_rs import read_calib_file
from openpose.native.python.utils_rs import read_color_file
from openpose.native.python.utils_rs import read_depth_file
from openpose.native.python.utils_rs import prepare_save_paths

from tracking.track import Tracker


def rs_extract_skeletons_and_track_offline(args: argparse.Namespace):
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

    display_speed = 1

    runtime = {'PE': [], 'TK': []}

    enable_timer = True

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
    while not dict_check(empty_dict) and not end_loop:

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
            tqdm_bar = tqdm(color_filepaths, dynamic_ncols=True)

            calib_file = os.path.dirname(os.path.dirname(color_filepaths[0]))
            calib_file = calib_file + "/calib/calib.csv"
            if os.path.exists(calib_file):
                intr_mat = read_calib_file(calib_file)
            else:
                intr_mat = None

            # 5. loop through filepaths of color image
            for idx, color_filepath in enumerate(tqdm_bar):

                if idx == 15:
                    runtime = {'PE': [], 'TK': []}

                # if _c < 480:
                #     _c += 1
                #     continue

                paths = color_filepath.split('/')
                depth_filepath = color_filepath.replace(paths[-2], 'depth')
                skel_filepath = color_filepath.replace(paths[-2], 'skeleton')
                skel_filepath = skel_filepath.replace(
                    os.path.splitext(skel_filepath)[1], '.csv')

                if TK is not None:
                    if delay_counter > 1:
                        delay_counter -= 1
                        _image = read_color_file(color_filepath)
                        TK.no_measurement_predict_and_update()
                        PE.display(dev=dev,
                                   speed=display_speed,
                                   scale=args.op_display,
                                   image=_image,
                                   bounding_box=True,
                                   tracks=TK.tracker.tracks)
                        continue
                    else:
                        delay_counter = delay_switch

                # 4. read and infer pose
                infer_out = extract_2dskeletons(args=args,
                                                pose_extractor=PE,
                                                color_src=color_filepath,
                                                depth_src=depth_filepath,
                                                intr_mat=intr_mat,
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

                # status = PE.display(dev=dev,
                #                     speed=display_speed,
                #                     scale=args.op_display,
                #                     image=_image,
                #                     bounding_box=False,
                #                     tracks=TK.tracks)

                if args.op_save_result_image and boxes is not None:
                    img = PE.pyop.opencv_image
                    if TK is not None:
                        img = PE.pyop._draw_skeleton_text_image(img, scores)
                        img = PE.pyop._draw_bounding_box_tracking_image(img, TK.tracks)  # noqa
                    else:
                        img = PE.pyop._draw_skeleton_text_image(img, scores)
                        img = PE.pyop._draw_skeleton_bounding_box_image(img, boxes)  # noqa
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

                # if args.op_rs_delete_image:
                #     os.remove(color_filepath)

            end_loop = True


class ExtractSkeletonAndTrack:

    def __init__(self, args, pose_extractor, tracker, enable_timer) -> None:

        self.args = args

        self.PE = pose_extractor
        self.TK = tracker

        self.enable_timer = enable_timer

        if args.op_proc:
            self.infer_queue = mp.Queue(maxsize=10)
            self.track_queue = mp.Queue(maxsize=10)
            self.write_queue = mp.Queue(maxsize=10)
            self.final_queue = mp.Queue(maxsize=10)
        else:
            self.infer_queue = Queue(maxsize=10)
            self.track_queue = Queue(maxsize=10)
            self.write_queue = Queue(maxsize=10)
            self.final_queue = Queue(maxsize=10)

    def start(self):
        if self.args.op_proc:
            self.p_extract = Thread(target=self.extract_skeletons, args=())
            self.p_track = mp.Process(target=self.track, args=())
            self.p_write = mp.Process(target=self.writeout, args=())
        else:
            self.p_extract = Thread(target=self.extract_skeletons, args=())
            self.p_track = Thread(target=self.track, args=())
            self.p_write = Thread(target=self.writeout, args=())
        self.p_extract.start()
        self.p_track.start()
        self.p_write.start()

    def queue_input(self, args):
        # args: filepaths
        self.infer_queue.put(args)

    def queue_output(self):
        return self.final_queue.get()

    def extract_skeletons(self):
        while True:
            (color_filepath, depth_filepath, skel_filepath,
             intr_mat, break_loop) = self.infer_queue.get()

            if break_loop:
                self.track_queue.put((None, None, None, None, None,
                                      None, None, None, None, break_loop))
                break

            infer_out = extract_2dskeletons(args=self.args,
                                            pose_extractor=self.PE,
                                            color_src=color_filepath,
                                            depth_src=depth_filepath,
                                            intr_mat=intr_mat,
                                            skeleton_filepath=skel_filepath,
                                            enable_timer=self.enable_timer)

            if self.PE.pyop.pose_empty:
                boxes, scores, keypoints, heatmaps, filered_skel = \
                    None, None, None, None, None
            else:
                boxes = self.PE.pyop.pose_bounding_box.copy()
                scores = self.PE.pyop.pose_scores.copy()
                keypoints = self.PE.pyop.pose_keypoints.copy()
                heatmaps = self.PE.pyop.pose_heatmaps.copy()
                filered_skel = self.PE.pyop.filtered_skel

            img = self.PE.pyop.opencv_image.copy()

            self.track_queue.put((img, boxes, scores, keypoints, heatmaps,
                                  filered_skel,
                                  infer_out['data_prep_time']+1e-8,
                                  infer_out['infer_time']+1e-8,
                                  color_filepath,
                                  break_loop))

    def track(self):
        while True:
            (img, boxes, scores, keypoints, heatmaps, filered_skel,
             prep_time, infer_time, color_fp, break_loop
             ) = self.track_queue.get()

            if break_loop:
                self.write_queue.put((None, None, None, None, None, None,
                                      break_loop))
                break

            if boxes is not None and self.TK is not None:
                with Timer("update", self.enable_timer, False) as t:
                    image_size = (self.args.op_rs_image_width,
                                  self.args.op_rs_image_height)
                    self.TK.update(boxes, scores, keypoints, heatmaps, image_size)  # noqa
                if self.args.op_save_result_image:
                    img = PYOP._draw_skeleton_text_image(img, scores)
                    img = PYOP._draw_bounding_box_tracking_image(img, self.TK.tracks)  # noqa
                duration = t.duration+1e-8
            else:
                if self.args.op_save_result_image:
                    img = PYOP._draw_skeleton_text_image(img, scores)
                    img = PYOP._draw_skeleton_bounding_box_image(img, boxes)  # noqa
                duration = -1

            self.write_queue.put((img, filered_skel, prep_time, infer_time,
                                  duration, color_fp, break_loop))

    def writeout(self):
        while True:
            (img, filered_skel, prep_time, infer_time, track_time,
             color_fp, break_loop) = self.write_queue.get()

            if img is None:
                self.final_queue.put((None, None, None, None, break_loop))
                break

            if self.args.op_save_result_image and filered_skel is not None:
                img = cv2.resize(img,
                                 (int(img.shape[1]*self.args.op_display),
                                  int(img.shape[0]*self.args.op_display)))
                paths = color_fp.split('/')
                save_file = color_fp.replace(
                    paths[-2], f'{paths[-2]}_{self.TK.name}')
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
                cv2.imwrite(save_file, img)

            self.final_queue.put(
                (filered_skel, prep_time, infer_time, track_time, break_loop))


def rs_extract_skeletons_and_track_offline_mp(args: argparse.Namespace):
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

    display_speed = 1

    runtime = {'PE': [], 'TK': []}

    enable_timer = True

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

    EST = ExtractSkeletonAndTrack(args, PE, TK, enable_timer)
    EST.start()

    # 1. If no error
    while not dict_check(empty_dict) and not end_loop:

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

            tqdm_bar = tqdm(color_filepaths, dynamic_ncols=True)
            data_len = len(color_filepaths)
            break_loop = False

            calib_file = os.path.dirname(os.path.dirname(color_filepaths[0]))
            calib_file = calib_file + "/calib/calib.csv"
            if os.path.exists(calib_file):
                intr_mat = read_calib_file(calib_file)
            else:
                intr_mat = None

            # 5. loop through filepaths of color image
            for idx, color_filepath in enumerate(tqdm_bar):

                if idx + 1 == data_len:
                    break_loop = True

                if idx == 15:
                    runtime = {'PE': [], 'TK': []}

                if _c < 590:
                    _c += 1
                    continue

                paths = color_filepath.split('/')
                depth_filepath = color_filepath.replace(paths[-2], 'depth')
                skel_filepath = color_filepath.replace(paths[-2], 'skeleton')
                skel_filepath = skel_filepath.replace(
                    os.path.splitext(skel_filepath)[1], '.csv')

                if TK is not None:
                    if delay_counter > 1:
                        delay_counter -= 1
                        _image = read_color_file(color_filepath)
                        TK.no_measurement_predict_and_update()
                        PE.display(dev=dev,
                                   speed=display_speed,
                                   scale=args.op_display,
                                   image=_image,
                                   bounding_box=True,
                                   tracks=TK.tracker.tracks)
                        continue
                    else:
                        delay_counter = delay_switch

                # 4. read and infer pose
                EST.queue_input(
                    (color_filepath, depth_filepath, skel_filepath,
                     intr_mat, break_loop))

                (filered_skel, prep_time, infer_time, track_time, break_loop
                 ) = EST.queue_output()

                if break_loop:
                    break

                # status = PE.display(dev=dev,
                #                     speed=display_speed,
                #                     scale=args.op_display,
                #                     image=_image,
                #                     bounding_box=False,
                #                     tracks=TK.tracks)

                runtime['PE'].append(1/infer_time)
                runtime['TK'].append(1/track_time)
                tqdm_bar.set_description(
                    f"Image : {color_filepath.split('/')[-1]} | "
                    f"#Skel filtered : {filered_skel} | "
                    f"#Tracks : {len(TK.tracks)} | "
                    f"Prep time : {prep_time:.3f} | "
                    f"Pose time : {infer_time:.3f} | "
                    f"Track time : {track_time:.3f} | "
                    f"FPS PE : {sum(runtime['PE'])/len(runtime['PE']):.3f} | "
                    f"FPS TK : {sum(runtime['TK'])/len(runtime['TK']):.3f}"
                )

                # if args.op_rs_delete_image:
                #     os.remove(color_filepath)

            end_loop = True


if __name__ == "__main__":

    [arg_op, _] = get_parser().parse_known_args()

    if arg_op.op_proc in ['sp', 'mp']:
        extract_skel_func = rs_extract_skeletons_and_track_offline_mp
    else:
        extract_skel_func = rs_extract_skeletons_and_track_offline

    # extract_skel_func(arg_op)

    arg_op.op_save_result_image = True

    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_deepsort")
    # arg_op.op_track_deepsort = True
    # extract_skel_func(arg_op)
    # arg_op.op_track_deepsort = False

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_bytetrack")
    arg_op.op_track_bytetrack = True
    extract_skel_func(arg_op)
    arg_op.op_track_bytetrack = False

    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_ocsort")
    # arg_op.op_track_ocsort = True
    # extract_skel_func(arg_op)
    # arg_op.op_track_ocsort = False

    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_strongsort")
    # arg_op.op_track_strongsort = True
    # extract_skel_func(arg_op)
    # arg_op.op_track_strongsort = False

    print(f"[INFO] : FINISHED")
