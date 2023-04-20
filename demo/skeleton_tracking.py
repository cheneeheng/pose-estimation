import os
from tqdm import tqdm

from openpose.native.python.inference_rs import get_parser
from openpose.native.python.inference_rs import OpenPosePoseExtractor
from openpose.native.python.inference_rs import Tracker
from openpose.native.python.inference_rs import ExtractSkeletonAndTrack
from openpose.native.python.inference_rs import read_calib_file
from openpose.native.python.inference_rs import get_rs_sensor_dir
from openpose.native.python.inference_rs import Error
from openpose.native.python.inference_rs import dict_check


if __name__ == "__main__":

    [args, _] = get_parser().parse_known_args()
    args.op_model_folder = "/usr/local/src/openpose/models/"
    args.op_model_pose = "BODY_25"
    args.op_net_resolution = "-1x368"
    args.op_skel_thres = 0.2
    args.op_max_true_body = 4
    args.op_heatmaps_add_parts = True
    args.op_heatmaps_add_bkg = True
    args.op_heatmaps_add_PAFs = True
    args.op_save_skel_folder = "./"
    args.op_save_skel = False
    args.op_save_skel_image = False
    # For skel extraction/tracking in inference.py
    args.op_input_color_image = ""
    args.op_image_width = 1920
    args.op_image_height = 1080
    # # For 3d skel extraction
    # args.op_patch_offset = 2
    # # For 3d skel extraction
    # args.op_ntu_format = False
    # args.op_extract_3d_skel = False
    # args.op_save_3d_skel = False
    args.op_display = 1.0
    # args.op_display_depth = 0  # Not used
    # For skel extraction/tracking in inference_rs.py
    args.op_rs_dir = "data/mot17"
    # args.op_rs_dir = "/data/realsense"
    args.op_rs_delete_image = False
    args.op_save_result_image = False
    args.op_proc = "sp"
    args.op_track_deepsort = True
    # args.op_track_bytetrack = True
    # args.op_track_ocsort = True
    # args.op_track_strongsort = True
    args.op_track_buffer = 30
    args.bytetracker_trackthresh = 0.2
    args.bytetracker_trackbuffer = 30
    args.bytetracker_matchthresh = 0.8
    args.ocsort_detthresh = 0.2
    print("========================================")
    print(">>>>> args <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")

    # # extract_skel_func = rs_extract_skeletons_and_track_offline_mp
    # # extract_skel_func(arg_op)
    # # arg_op.op_save_result_image = True
    # # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_deepsort")
    # # arg_op.op_track_deepsort = True
    # # extract_skel_func(arg_op)
    # # arg_op.op_track_deepsort = False
    # # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_bytetrack")
    # # arg_op.op_track_bytetrack = True
    # # extract_skel_func(arg_op)
    # # arg_op.op_track_bytetrack = False
    # # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_ocsort")
    # # arg_op.op_track_ocsort = True
    # # extract_skel_func(arg_op)
    # # arg_op.op_track_ocsort = False
    # # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_strongsort")
    # # arg_op.op_track_strongsort = True
    # # extract_skel_func(arg_op)
    # # arg_op.op_track_strongsort = False

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
    display_speed = 1000

    # Runtime logging
    enable_timer = True
    runtime = {'PE': [], 'TK': []}

    # Setup extract and track classes ------------------------------------------
    PoseExtractor = OpenPosePoseExtractor(args)
    PoseTracker = Tracker(args, 30//(delay_switch+1))
    EST = ExtractSkeletonAndTrack(
        args, PoseExtractor, PoseTracker, enable_timer)
    EST.start()

    # 1. If no error -----------------------------------------------------------
    while not dict_check(empty_dict) and not end_loop:

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
                    if empty_dict[dev].counter > 300:
                        print("[INFO] Retried 300 times and no new files...")
                        empty_dict[dev].state = True
                    continue

                filepath_dict[dev] += color_filepaths

        # 4. loop through devices for offline inference ------------------------
        for dev, color_filepaths in filepath_dict.items():

            if end_loop:
                break

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

            # 5. loop through filepaths of color image -------------------------
            for idx, color_filepath in enumerate(tqdm_bar):

                if idx + 1 == data_len:
                    break_loop = True

                if idx == 15:
                    runtime = {'PE': [], 'TK': []}

                # if _c < 550:
                #     _c += 1
                #     continue

                depth_filepath = color_filepath.replace(
                    color_filepath.split('/')[-2], 'depth')
                skel_filepath, skel_prefix = os.path.split(color_filepath)
                skel_filepath = os.path.join(
                    os.path.split(skel_filepath)[0], 'skeleton')
                skel_prefix = os.path.splitext(skel_prefix)[0]

                # 6. track without pose extraction -----------------------------
                if delay_counter > 0:
                    delay_counter -= 1
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
                    EST.queue_input((color_filepath, None, None, None, False))
                    # EST.queue_input(
                    #     (color_filepath, depth_filepath, skel_filepath,
                    #      intr_mat, False))

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

                    for track in EST.TK.tracks:
                        try:
                            # deepsort / ocsortq
                            bb = track.to_tlbr()
                        except AttributeError:
                            # bytetrack
                            bb = track.tlbr
                        print(track.track_id,
                              bb,
                              EST.PE.pyop.pose_bounding_box[track.det_id])

                    print(EST.PE.pyop.pose_bounding_box)
                    # print(EST.PE.pyop.pose_scores)
                    # EST.PE.pyop.pose_keypoints[EST.PE.pyop.pose_scores==track.score]

                    # 8. printout ----------------------------------------------
                    runtime['PE'].append(1/infer_time)
                    runtime['TK'].append(1/track_time)
                    tqdm_bar.set_description(
                        f"Image : {color_filepath.split('/')[-1]} | "
                        f"#Skel filtered : {filered_skel} | "
                        f"#Tracks : {len(EST.TK.tracks)} | "
                        f"Prep time : {prep_time:.3f} | "
                        f"Pose time : {infer_time:.3f} | "
                        f"Track time : {track_time:.3f} | "
                        f"FPS PE : {sum(runtime['PE'])/len(runtime['PE']):.3f} | "  # noqa
                        f"FPS TK : {sum(runtime['TK'])/len(runtime['TK']):.3f}"
                    )

                if break_loop:
                    EST.break_process_loops()
                    break

            end_loop = True
