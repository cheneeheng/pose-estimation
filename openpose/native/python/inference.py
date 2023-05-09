import argparse
import cv2
import os
import multiprocessing as mp
import numpy as np
from queue import Queue
from tqdm import tqdm
from threading import Thread
from typing import Optional, Union

from openpose.native.python.args import get_parser
from openpose.native.python.skeleton import OpenPosePoseExtractor
from openpose.native.python.skeleton import PyOpenPoseNative as PYOP
from openpose.native.python.utils import Timer
from openpose.native.python.utils_rs import read_color_file
from openpose.native.python.utils_rs import read_depth_file
from openpose.native.python.utils_rs import prepare_save_paths

from tracking.track import Tracker


class ExtractSkeletonAndTrack:

    def __init__(self, args, pose_extractor, tracker, enable_timer) -> None:

        self.args = args

        self.PE = pose_extractor
        self.TK = tracker

        self.enable_timer = enable_timer

        if args.op_proc == 'mp':
            raise ValueError("mp not supported for now, openpose inference "
                             "seems to cause the mp to hang.")
            # self.infer_queue = mp.Queue(maxsize=10)
            # self.track_queue = mp.Queue(maxsize=10)
            # self.write_queue = mp.Queue(maxsize=10)
            # self.final_queue = mp.Queue(maxsize=10)
        else:
            self.infer_queue = Queue(maxsize=10)
            self.track_queue = Queue(maxsize=10)
            self.write_queue = Queue(maxsize=10)
            self.final_queue = Queue(maxsize=10)

    def start(self):
        if self.args.op_proc == 'mp':
            raise ValueError("mp not supported for now, openpose inference "
                             "seems to cause the mp to hang.")
            # self.p_extract = mp.Process(
            #     target=self.extract_skeletons,
            #     args=(self.infer_queue, self.track_queue)
            # )
            # self.p_track = mp.Process(
            #     target=self.track,
            #     args=(self.track_queue, self.write_queue)
            # )
            # self.p_write = mp.Process(
            #     target=self.writeout,
            #     args=(self.write_queue, self.final_queue)
            # )
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

    def break_process_loops(self):
        self.queue_input((None, None, None, None, True))
        self.queue_output()

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
                                            skeleton_folder=skel_filepath,
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
                    image_size = (self.args.op_image_width,
                                  self.args.op_image_height)
                    self.TK.update(boxes, scores, keypoints, heatmaps, image_size)  # noqa
                if self.args.op_save_track_image:
                    img = PYOP._draw_text_on_skeleton_image(img, scores)
                    img = PYOP._draw_tracking_bounding_box_image(img, self.TK.tracks)  # noqa
                duration = t.duration+1e-8
            else:
                if self.args.op_save_track_image:
                    img = PYOP._draw_text_on_skeleton_image(img, scores)
                    img = PYOP._draw_bounding_box_on_skeleton_image(img, boxes)
                duration = -1

            self.write_queue.put((img, filered_skel, prep_time, infer_time,
                                  duration, color_fp, break_loop))

    def writeout(self):
        while True:
            (img, filered_skel, prep_time, infer_time, track_time,
             color_fp, break_loop) = self.write_queue.get()

            if break_loop:
                self.final_queue.put((None, None, None, None, break_loop))
                break

            if self.args.op_save_track_image and filered_skel is not None:
                paths = color_fp.split('/')
                save_file = color_fp.replace(
                    paths[-2], f'{paths[-2]}_{self.TK.name}')
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
                cv2.imwrite(save_file, img)

            self.final_queue.put(
                (filered_skel, prep_time, infer_time, track_time, break_loop))


def extract_2dskeletons(args: argparse.Namespace,
                        pose_extractor: OpenPosePoseExtractor,
                        color_src: Union[np.ndarray, str],
                        depth_src: Optional[Union[np.ndarray, str]] = None,
                        intr_mat: Optional[Union[np.ndarray, str]] = None,
                        skeleton_folder: Optional[str] = None,
                        skeleton_file_prefix: str = '',
                        enable_timer: bool = False) -> dict:
    """Extract pose using openpose on an image.

    Args:
        args (argparse.Namespace): inputs args.
        pose_extractor (OpenPosePoseExtractor): Openpose wrapper class.
        color_src (Union[np.ndarray, str]): rgb image or path to rgb image.
        depth_src (Optional[Union[np.ndarray, str]], optional): depth image or
            path to depth image. If None, the depth data will not be used.
            Defaults to None.
        skeleton_folder (Optional[str], optional): path to skeleton folder.
            If None, the extracted pose will not be saved. Defaults to None.
        skeleton_file_prefix (str): Prefix for the saved skeleton file.
        enable_timer (bool): If true printout timing. Defaults to False.

    Returns:
        bool: False if error, else True
        float: data_prep_time, -1 if error
        float: infer_time, -1 if error
    """
    assert color_src is not None

    with Timer(f"data preparation", enable_timer, printout=False) as t1:

        # 1. prepare save path
        save_path_dict = prepare_save_paths(skeleton_folder,
                                            skeleton_file_prefix,
                                            args.op_save_skel,
                                            args.op_save_skel_image,
                                            args.op_save_3d_skel)

        # 2a. get the color image
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

        # 2b. get the depth image
        depth = None
        if args.op_extract_3d_skel and depth_src is not None:
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

        # 3. reshape images
        try:
            img_h, img_w = args.op_image_height, args.op_image_width
            image = image.reshape(img_h, img_w, 3)
        except Exception as e:
            print(e)
            print(f"[WARN] : Error in reshaping image...")
            if isinstance(color_src, str):
                print(f"skipping {color_src}")
            return {'status': False,
                    'data_prep_time': -1,
                    'infer_time': -1}

    with Timer("infer", enable_timer, printout=False) as t2:

        # 4. infer images
        try:
            if args.op_extract_3d_skel:
                # main_dir = os.path.dirname(os.path.dirname(kpt_save_path))
                # intr_mat = read_calib_file(main_dir + "/calib/calib.csv")
                pose_extractor.predict_3d(
                    image,
                    depth,
                    intr_mat,
                    save_path_dict['skel_save_path'],
                    save_path_dict['3dskel_save_path'],
                    save_path_dict['skel_image_save_path']
                )
            else:
                pose_extractor.predict(
                    image,
                    depth,
                    save_path_dict['skel_save_path'],
                    save_path_dict['skel_image_save_path']
                )
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


def extract_2dskeletons_online(args: argparse.Namespace):
    """Runs openpose inference in online mode.

    Reads image files under the `color_image` arg and extracts pose
    from the images using openpose.

    Args:
        args (argparse.Namespace): CLI arguments
    """

    assert os.path.isdir(args.op_input_color_image), \
        f'{args.op_input_color_image} does not exist...'

    base_path = args.op_input_color_image

    # For cv.imshow
    display_speed = 1

    # Runtime logging
    enable_timer = True

    # 1. initialize class ------------------------------------------------------
    PE = OpenPosePoseExtractor(args)
    _c = 0
    break_loop = False

    # 2. loop through filepaths of color image ---------------------------------
    while not break_loop:

        filepaths = sorted([os.path.join(base_path, i)
                            for i in os.listdir(base_path)])

        for color_filepath in filepaths:

            _c += 1

            skel_filepath, _ = os.path.split(color_filepath)
            skel_filepath = os.path.join(os.path.split(skel_filepath)[0],
                                         'skeleton')

            # 3. infer pose ----------------------------------------------------
            infer_out = extract_2dskeletons(args=args,
                                            pose_extractor=PE,
                                            color_src=color_filepath,
                                            skeleton_folder=skel_filepath,
                                            enable_timer=enable_timer)

            status = PE.display(win_name='000',
                                speed=display_speed,
                                scale=args.op_display)

            if not status[0]:
                break_loop = True

            # 4. printout ------------------------------------------------------
            if _c % 100 == 0:
                infer_time = infer_out['infer_time']+1e-8
                prep_time = infer_out['data_prep_time']+1e-8
                filered_skel = PE.pyop.filtered_skel
                print(
                    f"Image : {color_filepath.split('/')[-1]} | "
                    f"#Skel filtered : {filered_skel} | "
                    f"Prep time : {prep_time:.3f} | "
                    f"Pose time : {infer_time:.3f}"
                )

            if break_loop:
                break


def extract_2dskeletons_offline(args: argparse.Namespace):
    """Runs openpose inference in offline mode.

    Reads image files under the `color_image` arg and extracts pose
    from the images using openpose.

    Args:
        args (argparse.Namespace): CLI arguments
    """

    assert os.path.isdir(args.op_input_color_image), \
        f'{args.op_input_color_image} does not exist...'

    base_path = args.op_input_color_image
    filepaths = sorted([os.path.join(base_path, i)
                        for i in os.listdir(base_path)])

    # For cv.imshow
    display_speed = 1

    # Runtime logging
    enable_timer = True
    runtime = {'PE': [], 'TK': []}

    # 1. initialize class ------------------------------------------------------
    PE = OpenPosePoseExtractor(args)

    # 2. loop through filepaths of color image ---------------------------------
    _c = 0
    tqdm_bar = tqdm(filepaths, dynamic_ncols=True)
    break_loop = False
    data_len = len(filepaths)
    for idx, color_filepath in enumerate(tqdm_bar):

        if idx + 1 == data_len:
            break_loop = True

        if idx == 15:
            runtime = {'PE': [], 'TK': []}

        skel_filepath, _ = os.path.split(color_filepath)
        skel_filepath = os.path.join(os.path.split(skel_filepath)[0],
                                     'skeleton')

        # 3. infer pose --------------------------------------------------------
        infer_out = extract_2dskeletons(args=args,
                                        pose_extractor=PE,
                                        color_src=color_filepath,
                                        skeleton_folder=skel_filepath,
                                        enable_timer=enable_timer)

        status = PE.display(win_name='000',
                            speed=display_speed,
                            scale=args.op_display)

        if not status[0]:
            break_loop = True

        # 4. printout ----------------------------------------------------------
        infer_time = infer_out['infer_time']+1e-8
        prep_time = infer_out['data_prep_time']+1e-8
        filered_skel = PE.pyop.filtered_skel
        runtime['PE'].append(1/infer_time)
        tqdm_bar.set_description(
            f"Image : {color_filepath.split('/')[-1]} | "
            f"#Skel filtered : {filered_skel} | "
            f"Prep time : {prep_time:.3f} | "
            f"Pose time : {infer_time:.3f} | "
            f"FPS PE : {sum(runtime['PE'])/len(runtime['PE']):.3f}"
        )

        if break_loop:
            break


def extract_2dskeletons_and_track_offline(args: argparse.Namespace):
    """Runs openpose inference and tracking in offline mode.

    Reads image files under the `color_image` arg and extracts pose
    from the images using openpose.

    Args:
        args (argparse.Namespace): CLI arguments
    """

    assert os.path.isdir(args.op_input_color_image), \
        f'{args.op_input_color_image} does not exist...'

    base_path = args.op_input_color_image
    filepaths = sorted([os.path.join(base_path, i)
                        for i in os.listdir(base_path)])

    # Delay = predict and no update in tracker
    delay_switch = 0
    delay_counter = 0

    # For cv.imshow
    display_speed = 1

    # Runtime logging
    enable_timer = True
    runtime = {'PE': [], 'TK': []}

    # Setup extract and track classes ------------------------------------------
    PoseExtractor = OpenPosePoseExtractor(args)
    PoseTracker = Tracker(args, 30//(delay_switch+1))
    EST = ExtractSkeletonAndTrack(
        args, PoseExtractor, PoseTracker, enable_timer)
    EST.start()

    # 1. loop through devices --------------------------------------------------
    _c = 0
    tqdm_bar = tqdm(filepaths, dynamic_ncols=True)
    break_loop = False
    data_len = len(filepaths)

    # 2. loop through filepaths of color image ---------------------------------
    for idx, color_filepath in enumerate(tqdm_bar):

        if idx + 1 == data_len:
            break_loop = True

        if idx == 15:
            runtime = {'PE': [], 'TK': []}

        # if _c < 480:
        #     _c += 1
        #     continue

        skel_filepath, _ = os.path.split(color_filepath)
        skel_filepath = os.path.join(os.path.split(skel_filepath)[0],
                                     'skeleton')

        # 3. track without pose extraction -----------------------------
        if delay_counter > 0:
            delay_counter -= 1
            _image = read_color_file(color_filepath)
            EST.TK.no_measurement_predict_and_update()
            EST.PE.display(dev='000',
                           speed=display_speed,
                           scale=args.op_display,
                           image=_image,
                           bounding_box=True,
                           tracks=EST.TK.tracker.tracks)

        else:
            delay_counter = delay_switch

            # 4. infer pose and track ------------------------------------------
            EST.queue_input((color_filepath, None, skel_filepath, None, False))

            (filered_skel, prep_time, infer_time, track_time, _
             ) = EST.queue_output()

            status = EST.PE.display(win_name='000',
                                    speed=display_speed,
                                    scale=args.op_display,
                                    image=None,
                                    bounding_box=False,
                                    tracks=EST.TK.tracks)
            if not status[0]:
                break_loop = True

            # 5. printout ------------------------------------------------------
            runtime['PE'].append(1/infer_time)
            runtime['TK'].append(1/track_time)
            tqdm_bar.set_description(
                f"Image : {color_filepath.split('/')[-1]} | "
                f"#Skel filtered : {filered_skel} | "
                f"#Tracks : {len(EST.TK.tracks)} | "
                f"Prep time : {prep_time:.3f} | "
                f"Pose time : {infer_time:.3f} | "
                f"Track time : {track_time:.3f} | "
                f"FPS PE : {sum(runtime['PE'])/len(runtime['PE']):.3f} | "
                f"FPS TK : {sum(runtime['TK'])/len(runtime['TK']):.3f}"
            )

        if break_loop:
            EST.break_process_loops()
            break


def compare_tracker(arg_op: argparse.Namespace):

    extract_skel_func = extract_2dskeletons_and_track_offline

    arg_op.op_input_color_image = "data/mot17/dev1/100/color"
    arg_op.op_save_track_image = False

    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_deepsort")
    # arg_op.op_track_deepsort = True
    # arg_op.op_heatmaps_add_PAFs = True
    # arg_op.op_heatmaps_add_bkg = True
    # arg_op.op_heatmaps_add_parts = True
    # extract_skel_func(arg_op)
    # arg_op.op_track_deepsort = False
    # arg_op.op_heatmaps_add_PAFs = False
    # arg_op.op_heatmaps_add_bkg = False
    # arg_op.op_heatmaps_add_parts = False

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
    # arg_op.op_heatmaps_add_PAFs = True
    # arg_op.op_heatmaps_add_bkg = True
    # arg_op.op_heatmaps_add_parts = True
    # extract_skel_func(arg_op)
    # arg_op.op_track_strongsort = False
    # arg_op.op_heatmaps_add_PAFs = False
    # arg_op.op_heatmaps_add_bkg = False
    # arg_op.op_heatmaps_add_parts = False


if __name__ == "__main__":

    arg_op, _ = get_parser().parse_known_args()

    if (arg_op.op_track_deepsort or arg_op.op_track_bytetrack or
            arg_op.op_track_ocsort or arg_op.op_track_strongsort):
        extract_2dskeletons_and_track_offline(arg_op)
    elif arg_op.op_mode == 'single':
        PE = OpenPosePoseExtractor(arg_op)
        enable_timer = True
        extract_2dskeletons(args=arg_op,
                            pose_extractor=PE,
                            color_src=arg_op.op_input_color_image,
                            skeleton_folder=arg_op.op_save_skel_folder,
                            enable_timer=enable_timer)
    elif arg_op.op_mode == 'online':
        extract_2dskeletons_online(arg_op)
    elif arg_op.op_mode == 'offline':
        extract_2dskeletons_offline(arg_op)
    elif arg_op.op_mode == 'comparetracker':
        compare_tracker(arg_op)
    else:
        raise ValueError("Unknown op_mode")

    print(f"[INFO] : FINISHED")
