import json
import numpy as np
import os
import time
import sys
from functools import partial

import torch
from torch.nn import functional as F

from rs_py.rs_run_devices import get_rs_parser
from rs_py.rs_run_devices import printout
from rs_py.rs_run_devices import RealsenseWrapper

from openpose.native.python.inference_rs import get_parser as get_op_parser
from openpose.native.python.inference_rs import OpenPosePoseExtractor
from openpose.native.python.inference_rs import Tracker
from openpose.native.python.inference_rs import ExtractSkeletonAndTrack

from infer.inference import ActionRecognition
from utils.parser import get_parser as get_ar_parser
from utils.parser import load_parser_args_from_config
from utils.utils import init_seed


def get_rs_args():
    args, _ = get_rs_parser().parse_known_args()
    args.rs_steps = 1000
    args.rs_fps = 30
    args.rs_ir_emitter_power = 150
    args.rs_display_frame = 1
    args.rs_save_with_key = False
    args.rs_save_data = False
    args.rs_save_path = ''
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
    # args.op_heatmaps_add_parts = True
    # args.op_heatmaps_add_bkg = True
    # args.op_heatmaps_add_PAFs = True
    args.op_save_skel_folder = ""
    args.op_save_skel = False
    args.op_save_skel_image = False
    # For skel extraction/tracking in inference.py
    args.op_input_color_image = ""
    args.op_image_width = 848
    args.op_image_height = 480
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
    args.op_rs_delete_image = False
    args.op_save_result_image = False
    args.op_proc = "sp"
    # args.op_track_deepsort = True
    args.op_track_bytetrack = True
    # args.op_track_ocsort = True
    # args.op_track_strongsort = True
    args.op_track_buffer = 30
    args.bytetracker_trackthresh = 0.2
    args.bytetracker_trackbuffer = 30
    args.bytetracker_matchthresh = 0.8
    args.bytetracker_mot20 = False
    print("========================================")
    print(">>>>> op_args <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")
    return args


def get_ar_args():
    init_seed(1)
    # label_mapping_file = "/data/07_AAGCN/model/ntu_15j/index_to_name.json"
    label_mapping_file = "/data/07_AAGCN/model/ntu_15j_9l/index_to_name.json"
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/weight/SGN-116-68208.pt"  # noqa
    weights = "/data/07_AAGCN/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/weight/SGN-110-10670.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_ntu_result/xview/aagcn_joint/230414100001/weight/Model-50-29400.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_joint/230414100001/weight/Model-50-4850.pt"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/config.yaml"  # noqa
    config = "/data/07_AAGCN/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_ntu_result/xview/aagcn_joint/230414100001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_joint/230414100001/config.yaml"  # noqa
    parser = get_ar_parser()
    parser.set_defaults(**{'config': config})
    args = load_parser_args_from_config(parser)
    args.weights = weights
    args.max_frame = 300
    args.max_num_skeleton_true = 2
    args.max_num_skeleton = 4
    args.num_joint = 15
    args.gpu = True
    args.timing = False
    args.interval = 0
    args.moving_avg = 1
    args.aagcn_normalize = True
    args.sgn_preprocess = True
    args.multi_test = 5
    # args.data_path = data_path
    args.label_mapping_file = label_mapping_file
    # args.out_folder = "/data/07_AAGCN/data_tmp/delme"
    print("========================================")
    print(">>>>> ar_args <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")
    return args


class SkeletonStorage():
    def __init__(self) -> None:
        self.ids = []
        self.skeletons = []
        self.counter = []
        self.valid = []

    def add(self, track_id, keypoints):
        if track_id in self.ids:
            sid = self.ids.index(track_id)
            self.skeletons[sid].append(keypoints)
            self.valid.append(True)
        else:
            self.ids.append(track_id)
            self.skeletons.append([keypoints])
            self.counter.append(0)
            self.valid.append(True)

    def get_last_skel(self):
        skels = []
        for skeleton in self.skeletons:
            skels.append(skeleton[-1])
        return np.expand_dims(np.stack(skels, 0), 1)  # M, 1, V, C

    def delete(self, idx):
        self.ids.pop(idx)
        self.skeletons.pop(idx)
        self.counter.pop(idx)
        self.valid.pop(idx)

    def check_valid_and_delete(self):
        num_data = len(self.ids)
        del_idx = []
        for i in range(num_data):
            if not self.valid[i]:
                self.counter[i] += 1
            if self.counter[i] > 30:
                del_idx.append(self.ids[i])
            self.valid[i] = False
        for i in del_idx:
            self.delete(i)


if __name__ == "__main__":

    # Delay = predict and no update in tracker
    delay_switch = 0
    delay_counter = 0

    # For cv.imshow
    display_speed = 1

    # Runtime logging
    enable_timer = True
    runtime = {'PE': [], 'TK': []}

    # For skeleton storage
    skelstorage = SkeletonStorage()

    # 1. args ------------------------------------------------------------------
    rs_args = get_rs_args()
    op_args = get_op_args()
    ar_args = get_ar_args()

    # 2. setup realsense -------------------------------------------------------
    rsw = RealsenseWrapper(rs_args, rs_args.rs_dev)
    rsw.initialize_depth_sensor_ae()

    if len(rsw.enabled_devices) == 0:
        raise ValueError("no devices connected")

    rsw = RealsenseWrapper(rs_args, rs_args.rs_dev)
    rsw.initialize()

    rsw.flush_frames(rs_args.rs_fps * 5)
    time.sleep(3)

    device_sn = list(rsw.enabled_devices.keys())[0]

    # 3. Setup extract and track classes ---------------------------------------
    PoseExtractor = OpenPosePoseExtractor(op_args)
    PoseTracker = Tracker(op_args, 30//(delay_switch+1))
    EST = ExtractSkeletonAndTrack(
        op_args, PoseExtractor, PoseTracker, enable_timer)
    EST.start()

    # 4. setup aagcn -----------------------------------------------------------
    with open(ar_args.label_mapping_file, 'r') as f:
        MAPPING = {int(i): j for i, j in json.load(f).items()}

    AR = ActionRecognition(ar_args)

    ar_start_time = time.time()

    # MAIN LOOP ----------------------------------------------------------------
    try:
        c = 0
        max_c = int(1e8)

        while True:

            if c < 15:
                runtime = {'PE': [], 'TK': []}

            # 4. grab image from realsense -------------------------------------
            rsw.step(
                display=rs_args.rs_display_frame,
                display_and_save_with_key=rs_args.rs_save_with_key
            )
            color_image = rsw.frames[device_sn]['color_framedata']  # h,w,c
            depth_image = rsw.frames[device_sn]['depth_framedata']  # h,w

            if rsw.key & 0xFF == ord('q'):
                printout("`q` button pressed", 'i')
                break

            # 5. track without pose extraction ---------------------------------
            if delay_counter > 0:
                delay_counter -= 1
                EST.TK.no_measurement_predict_and_update()
                EST.PE.display(win_name="000",
                               speed=display_speed,
                               scale=op_args.op_display,
                               image=None,
                               bounding_box=True,
                               tracks=EST.TK.tracks)
            else:
                delay_counter = delay_switch

                # 6. infer pose and track --------------------------------------
                EST.queue_input((color_image, None, None, None, False))

                (filered_skel, prep_time, infer_time, track_time, _
                 ) = EST.queue_output()

                for track in EST.TK.tracks:
                    skelstorage.add(track.track_id,
                                    EST.PE.pyop.pose_keypoints[track.det_id])
                skelstorage.check_valid_and_delete()

                status = EST.PE.display(win_name="000",
                                        speed=display_speed,
                                        scale=op_args.op_display,
                                        image=None,
                                        bounding_box=False,
                                        tracks=EST.TK.tracks)
                if not status[0]:
                    break

            # 7. action recognition --------------------------------------------
            if time.time() - ar_start_time > int(ar_args.interval):
                if enable_timer:
                    start_time = time.time()

                if len(skelstorage.skeletons) > 0:
                    # data  # M, 1, V, C
                    AR.append_data(skelstorage.get_last_skel())

                    # logits = [#labels], prediction = label from 0 - #labels-1
                    logits, prediction = AR.predict()
                    print(MAPPING[prediction+1])

                if enable_timer:
                    recog_time = -1
                else:
                    recog_time = time.time() - start_time

            # 8. printout ------------------------------------------------------
            if c % rs_args.rs_fps == 0:
                # printout(
                #     f"Step {c:12d} :: "
                #     f"{[i.get('color_timestamp', None) for i in rsw.frames.values()]} :: "  # noqa
                #     f"{[i.get('depth_timestamp', None) for i in rsw.frames.values()]}",  # noqa
                #     'i'
                # )
                color_timestamp = list(rsw.frames.values())[0].get('color_timestamp')  # noqa
                runtime['PE'].append(1/infer_time)
                runtime['TK'].append(1/track_time)
                printout(
                    f"Image : {color_timestamp} | "
                    f"#Skel filtered : {filered_skel} | "
                    f"#Tracks : {len(EST.TK.tracks)} | "
                    f"Prep time : {prep_time:.3f} | "
                    f"Pose time : {infer_time:.3f} | "
                    f"Track time : {track_time:.3f} | "
                    f"Recog time : {recog_time:.3f} | "
                    f"FPS PE : {sum(runtime['PE'])/len(runtime['PE']):.3f} | "
                    f"FPS TK : {sum(runtime['TK'])/len(runtime['TK']):.3f}",
                    'i'
                )

            if not len(rsw.frames) > 0:
                printout(f"Empty...", 'w')
                continue

            c += 1
            if c > rs_args.rs_fps * rs_args.rs_steps or c > max_c:
                break

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        printout(f"{e}", 'e')
        printout(f"Stopping RealSense devices...", 'i')
        rsw.stop()

    finally:
        printout(f"Final RealSense devices...", 'i')
        rsw.stop()

    EST.break_process_loops()

    printout(f"Finished...", 'i')
