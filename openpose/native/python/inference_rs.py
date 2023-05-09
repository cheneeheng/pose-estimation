import argparse
import os
from tqdm import tqdm
import time

from openpose.native.python.args import get_parser
from openpose.native.python.inference import extract_2dskeletons
from openpose.native.python.inference import ExtractSkeletonAndTrack
from openpose.native.python.skeleton import OpenPosePoseExtractor
from openpose.native.python.utils import dict_check
from openpose.native.python.utils import Error
from openpose.native.python.utils_rs import get_rs_sensor_dir
from openpose.native.python.utils_rs import read_calib_file
from openpose.native.python.utils_rs import read_color_file

from tracking.track import Tracker


def rs_extract_skeletons_online_mp(args: argparse.Namespace):
    """Runs openpose inference on realsense camera in online mode.

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

    # Delay = predict and no update in tracker
    delay_switch = 0
    delay_counter = 0

    # For cv.imshow
    display_speed = 1

    # Runtime logging
    enable_timer = True
    runtime = {'PE': [], 'TK': []}

    # Setup extract and track classes ------------------------------------------
    PE = OpenPosePoseExtractor(args)
    _c = 0

    # 1. If no error -----------------------------------------------------------
    while not end_loop:

        filepath_dict = {i: [] for i in dev_list}

        # 2. loop through devices ----------------------------------------------
        for dev, trial_color_dir in dev_trial_color_dir.items():

            # 3. loop through trials -------------------------------------------
            for trial, color_dir in trial_color_dir.items():

                color_filepaths = [os.path.join(color_dir, i)
                                   for i in sorted(os.listdir(color_dir))]

                if len(color_filepaths) == 0:
                    print(f"[INFO] : {color_dir} is empty...")
                    empty_dict[dev].counter += 1
                    time.sleep(1)
                    if empty_dict[dev].counter > 300:
                        print(f"[INFO] Retried 300s for {dev} and no new files...")  # noqa
                        # empty_dict[dev].state = True
                        end_loop = True
                    continue

                filepath_dict[dev] += color_filepaths

        _c += 1

        # 4. loop through devices for offline inference ------------------------
        for dev, color_filepaths in filepath_dict.items():

            if len(color_filepaths) == 0:
                continue

            break_loop = False

            # 5. loop through filepaths of color image -------------------------
            for color_filepath in color_filepaths:

                skel_filepath, _ = os.path.split(color_filepath)
                skel_filepath = os.path.join(
                    os.path.split(skel_filepath)[0], 'skeleton')

                # 6. infer pose ------------------------------------------------
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

                # 7. printout --------------------------------------------------
                if _c % 100 == 0:
                    infer_time = infer_out['infer_time']+1e-8
                    prep_time = infer_out['data_prep_time']+1e-8
                    filered_skel = PE.pyop.filtered_skel
                    runtime['PE'].append(1/infer_time)
                    print(
                        f"Image : {color_filepath.split('/')[-1]} | "
                        f"#Skel filtered : {filered_skel} | "
                        f"Prep time : {prep_time:.3f} | "
                        f"Pose time : {infer_time:.3f} | "
                        f"FPS PE : {sum(runtime['PE'])/len(runtime['PE']):.3f}"
                    )

                if args.op_rs_delete_image:
                    os.remove(color_filepath)

                if break_loop:
                    break

            if break_loop:
                end_loop = True
                break


def rs_extract_skeletons_offline_mp(args: argparse.Namespace):
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

    # For cv.imshow
    display_speed = 1

    # Runtime logging
    enable_timer = True
    runtime = {'PE': [], 'TK': []}

    # 1. Setup extract and track classes ---------------------------------------
    PE = OpenPosePoseExtractor(args)
    filepath_dict = {i: [] for i in dev_list}

    # 2. loop through devices --------------------------------------------------
    # 3. loop through trials ---------------------------------------------------
    for dev, trial_color_dir in dev_trial_color_dir.items():
        for trial, color_dir in trial_color_dir.items():
            color_filepaths = [os.path.join(color_dir, i)
                               for i in sorted(os.listdir(color_dir))]
            filepath_dict[dev] += color_filepaths

    # 4. loop through devices for offline inference ----------------------------
    for dev, color_filepaths in filepath_dict.items():

        if len(color_filepaths) == 0:
            raise ValueError("[ERRO] : no color images are detected")

        tqdm_bar = tqdm(color_filepaths, dynamic_ncols=True)
        data_len = len(color_filepaths)
        break_loop = False

        # 5. loop through filepaths of color image -----------------------------
        for idx, color_filepath in enumerate(tqdm_bar):

            if idx + 1 == data_len:
                break_loop = True

            if idx == 15:
                runtime = {'PE': [], 'TK': []}

            skel_filepath, _ = os.path.split(color_filepath)
            skel_filepath = os.path.join(
                os.path.split(skel_filepath)[0], 'skeleton')

            # 6. infer pose ----------------------------------------------------
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

            # 7. printout ------------------------------------------------------
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

            if args.op_rs_delete_image:
                os.remove(color_filepath)

            if break_loop:
                break

        if break_loop:
            break


def rs_extract_skeletons_and_track_online_mp(args: argparse.Namespace):
    """Runs openpose inference and tracking on realsense camera in online mode.

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
    _c = 0

    # 1. If no error -----------------------------------------------------------
    while not end_loop:

        filepath_dict = {i: [] for i in dev_list}

        # 2. loop through devices ----------------------------------------------
        for dev, trial_color_dir in dev_trial_color_dir.items():

            # 3. loop through trials -------------------------------------------
            for trial, color_dir in trial_color_dir.items():

                color_filepaths = [os.path.join(color_dir, i)
                                   for i in sorted(os.listdir(color_dir))]

                if len(color_filepaths) == 0:
                    print(f"[INFO] : {color_dir} is empty...")
                    empty_dict[dev].counter += 1
                    time.sleep(1)
                    if empty_dict[dev].counter > 300:
                        print(f"[INFO] Retried 300s for {dev} and no new files...")  # noqa
                        # empty_dict[dev].state = True
                        end_loop = True
                    continue

                filepath_dict[dev] += color_filepaths

        _c += 1

        # 4. loop through devices for offline inference ------------------------
        for dev, color_filepaths in filepath_dict.items():

            if len(color_filepaths) == 0:
                continue

            break_loop = False

            calib_file = os.path.dirname(os.path.dirname(color_filepaths[0]))
            calib_file = calib_file + "/calib/calib.csv"
            if os.path.exists(calib_file):
                intr_mat = read_calib_file(calib_file)
            else:
                intr_mat = None

            # 5. loop through filepaths of color image -------------------------
            for color_filepath in color_filepaths:

                depth_filepath = color_filepath.replace(
                    color_filepath.split('/')[-2], 'depth')
                skel_filepath, _ = os.path.split(color_filepath)
                skel_filepath = os.path.join(
                    os.path.split(skel_filepath)[0], 'skeleton')

                # 6. track without pose extraction -----------------------------
                if delay_counter > 0:
                    delay_counter -= 1
                    _image = read_color_file(color_filepath)
                    EST.TK.no_measurement_predict_and_update()
                    EST.PE.display(win_name=dev,
                                   speed=display_speed,
                                   scale=args.op_display,
                                   image=None,
                                   bounding_box=True,
                                   tracks=EST.TK.tracks)

                else:
                    delay_counter = delay_switch

                    # 7. infer pose and track ----------------------------------
                    EST.queue_input(
                        (color_filepath, depth_filepath, skel_filepath,
                         intr_mat, False))

                    (filered_skel, prep_time, infer_time, track_time, _
                     ) = EST.queue_output()

                    status = EST.PE.display(win_name=dev,
                                            speed=display_speed,
                                            scale=args.op_display,
                                            image=None,
                                            bounding_box=False,
                                            tracks=EST.TK.tracks)
                    if not status[0]:
                        break_loop = True

                    # 8. printout ----------------------------------------------
                    if _c % 100 == 0:
                        runtime['PE'].append(1/infer_time)
                        runtime['TK'].append(1/track_time)
                        print(
                            f"Image : {color_filepath.split('/')[-1]} | "
                            f"#Skel filtered : {filered_skel} | "
                            f"#Tracks : {len(EST.TK.tracks)} | "
                            f"Prep time : {prep_time:.3f} | "
                            f"Pose time : {infer_time:.3f} | "
                            f"Track time : {track_time:.3f} | "
                            f"FPS PE : {sum(runtime['PE'])/len(runtime['PE']):.3f} | "  # noqa
                            f"FPS TK : {sum(runtime['TK'])/len(runtime['TK']):.3f}"  # noqa
                        )

                    if args.op_rs_delete_image:
                        os.remove(color_filepath)

                if break_loop:
                    EST.break_process_loops()
                    break

            if break_loop:
                end_loop = True
                break


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

    # Delay = predict and no update in tracker
    delay_switch = 0
    delay_counter = 0

    # For cv.imshow
    display_speed = 1

    # Runtime logging
    enable_timer = True
    runtime = {'PE': [], 'TK': []}

    # 1. Setup extract and track classes ---------------------------------------
    PoseExtractor = OpenPosePoseExtractor(args)
    PoseTracker = Tracker(args, 30//(delay_switch+1))
    EST = ExtractSkeletonAndTrack(
        args, PoseExtractor, PoseTracker, enable_timer)
    EST.start()

    filepath_dict = {i: [] for i in dev_list}

    # 2. loop through devices --------------------------------------------------
    # 3. loop through trials ---------------------------------------------------
    for dev, trial_color_dir in dev_trial_color_dir.items():
        for trial, color_dir in trial_color_dir.items():
            color_filepaths = [os.path.join(color_dir, i)
                               for i in sorted(os.listdir(color_dir))]
            filepath_dict[dev] += color_filepaths

    # 4. loop through devices for offline inference ----------------------------
    for dev, color_filepaths in filepath_dict.items():

        if len(color_filepaths) == 0:
            raise ValueError("[ERRO] : no color images are detected")

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

        # 5. loop through filepaths of color image -----------------------------
        for idx, color_filepath in enumerate(tqdm_bar):

            if idx + 1 == data_len:
                break_loop = True

            if idx == 15:
                runtime = {'PE': [], 'TK': []}

            # if _c < 590:
            #     _c += 1
            #     continue

            depth_filepath = color_filepath.replace(
                color_filepath.split('/')[-2], 'depth')
            skel_filepath, _ = os.path.split(color_filepath)
            skel_filepath = os.path.join(
                os.path.split(skel_filepath)[0], 'skeleton')

            # 6. track without pose extraction ---------------------------------
            if delay_counter > 0:
                delay_counter -= 1
                _image = read_color_file(color_filepath)
                EST.TK.no_measurement_predict_and_update()
                EST.PE.display(win_name=dev,
                               speed=display_speed,
                               scale=args.op_display,
                               image=None,
                               bounding_box=True,
                               tracks=EST.TK.tracks)

            else:
                delay_counter = delay_switch

                # 7. infer pose and track --------------------------------------
                EST.queue_input(
                    (color_filepath, depth_filepath, skel_filepath,
                        intr_mat, False))

                (filered_skel, prep_time, infer_time, track_time, _
                 ) = EST.queue_output()

                status = EST.PE.display(win_name=dev,
                                        speed=display_speed,
                                        scale=args.op_display,
                                        image=None,
                                        bounding_box=False,
                                        tracks=EST.TK.tracks)
                if not status[0]:
                    break_loop = True

                # 8. printout --------------------------------------------------
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

                if args.op_rs_delete_image:
                    os.remove(color_filepath)

            if break_loop:
                EST.break_process_loops()
                break

        if break_loop:
            break


if __name__ == "__main__":

    arg_op, _ = get_parser().parse_known_args()

    if (arg_op.op_track_deepsort or arg_op.op_track_bytetrack or
            arg_op.op_track_ocsort or arg_op.op_track_strongsort):
        if arg_op.op_mode == 'online':
            rs_extract_skeletons_and_track_online_mp(arg_op)
        elif arg_op.op_mode == 'offline':
            rs_extract_skeletons_and_track_offline_mp(arg_op)
        else:
            raise ValueError("Unknown mode")
    elif arg_op.op_mode == 'online':
        rs_extract_skeletons_online_mp(arg_op)
    elif arg_op.op_mode == 'offline':
        rs_extract_skeletons_offline_mp(arg_op)
    else:
        raise ValueError("Unknown op_mode")

    print(f"[INFO] : FINISHED")
