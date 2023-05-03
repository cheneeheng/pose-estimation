import cv2
import json
import numpy as np
import os
import time
import sys
from functools import partial
import traceback
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

from rs_py import rs
from rs_py.utility import get_filepaths_with_timestamps
from rs_py.utility import read_calib_file
from rs_py.utility import read_color_file
from rs_py.utility import read_depth_file
from rs_py.rs_run_devices import get_rs_parser
from rs_py.rs_run_devices import printout
from rs_py.rs_run_devices import RealsenseWrapper

from openpose.native.python.inference_rs import get_parser as get_op_parser
from openpose.native.python.inference_rs import OpenPosePoseExtractor
from openpose.native.python.inference_rs import Tracker
from openpose.native.python.inference_rs import ExtractSkeletonAndTrack
from openpose.native.python.utils import Timer

from data_gen.openpose.openpose_b25_j15_ntu_gendata import joint_mapping
from infer.inference import ActionRecognition
from utils.parser import get_parser as get_ar_parser
from utils.parser import load_parser_args_from_config
from utils.utils import init_seed
from utils.visualization import visualize_3dskeleton_in_matplotlib
from utils.visualization import visualize_3dskeleton_in_matplotlib_step


def get_rs_args():
    args, _ = get_rs_parser().parse_known_args()
    args.rs_steps = 1000
    args.rs_fps = 30
    args.rs_ir_emitter_power = 290
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
    args.op_patch_offset = 3
    # # For 3d skel extraction
    # args.op_extract_3d_skel = False
    # args.op_save_3d_skel = False
    args.op_display = 1.0
    # args.op_display_depth = 0  # Not used
    # For skel extraction/tracking in inference_rs.py
    args.op_rs_dir = "data/mot17"
    args.op_rs_delete_image = False
    args.op_save_track_image = False
    args.op_proc = "sp"
    # args.op_track_deepsort = True
    args.op_track_bytetrack = True
    # args.op_track_ocsort = True
    # args.op_track_strongsort = True
    args.op_track_buffer = 30
    args.bytetracker_trackthresh = 0.25
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
    args.max_frame = 100
    args.max_num_skeleton_true = 2
    args.max_num_skeleton = 4
    args.num_joint = 15
    args.gpu = True
    args.timing = False
    args.interval = 0
    args.moving_avg = 3
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
        self.skeletons = []  # M, T, V, C
        self.scores = []  # M, T, V
        self.counter = []
        self.valid = []

    def add(self, track_id, keypoints, scores):
        if track_id in self.ids:
            sid = self.ids.index(track_id)
            self.skeletons[sid].append(keypoints)
            self.scores[sid].append(scores)
            self.valid[sid] = True
        else:
            self.ids.append(track_id)
            self.skeletons.append([keypoints])
            self.scores.append([scores])
            self.counter.append(0)
            self.valid.append(True)

    def get_last_skel(self):
        skeletons = []
        scores = []
        for skeleton, score in zip(self.skeletons, self.scores):
            skeletons.append(skeleton[-1])
            scores.append(score[-1])
        skeleton = np.expand_dims(np.stack(skeletons, 0), 1)  # M, 1, V, C
        score = np.expand_dims(np.stack(scores, 0), 1)  # M, 1, V
        return skeleton, score

    def delete(self, idx):
        self.ids.pop(idx)
        self.skeletons.pop(idx)
        self.scores.pop(idx)
        self.counter.pop(idx)
        self.valid.pop(idx)

    def check_valid_and_delete(self):
        num_data = len(self.counter)
        del_idx = []
        for i in range(num_data):
            if not self.valid[i]:
                self.counter[i] += 1
            if self.counter[i] > 30:
                del_idx.append(i)
            self.valid[i] = False
        for i in del_idx:
            self.delete(i)


def visualize_rs(color_image: np.ndarray, depth_colormap: np.ndarray,
                 depth_scale: float, winname: str):
    # overlay depth colormap with color image
    images_overlapped = cv2.addWeighted(src1=color_image, alpha=0.3,
                                        src2=depth_colormap, beta=0.5,
                                        gamma=0)
    # Set pixels further than clipping_distance to grey
    clipping_distance = 3.5 / depth_scale
    grey_color = 153
    # depth image is 1 channel, color is 3 channels
    depth_image_3d = np.dstack(
        (depth_image, depth_image, depth_image))
    bg_removed = np.where(
        (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
        grey_color, images_overlapped)
    cv2.namedWindow(f'{winname}', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(f'{winname}', bg_removed)
    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        cv2.waitKey(5)
        printout("`q` button pressed", 'i')
        return True
    else:
        return False


def transform_3dpose(joints3d, depth_scale):
    matrix_z = np.array(
        [[0.99911548, -0.03845016, -0.01702447],
         [0.03845016, 0.67144157, 0.74005933],
         [-0.01702447, -0.74005933, 0.67232608]]
    )
    joints3d += np.array([0, -3000*depth_scale, 500*depth_scale])
    joints3d = (matrix_z@joints3d.transpose()).transpose()
    joints3d[:, 0] *= -0.7
    joints3d[:, 1] *= -1
    return joints3d


if __name__ == "__main__":

    motion = []

    # Delay = predict and no update in tracker
    delay_switch = 0
    delay_counter = 0

    # For cv.imshow
    display_speed = 1

    # Runtime logging
    enable_timer = True
    fps = {'PE': [], 'TK': []}
    infer_time = -1
    track_time = -1
    filered_skel = -1
    prep_time = -1
    recog_time = -1
    rs_time = -1

    # For skeleton storage
    skelstorage = SkeletonStorage()

    # 1. args ------------------------------------------------------------------
    rs_args = get_rs_args()
    op_args = get_op_args()
    ar_args = get_ar_args()

    # 2. setup realsense -------------------------------------------------------
    # rsw = RealsenseWrapper(rs_args, rs_args.rs_dev)
    # rsw.initialize_depth_sensor_ae()

    # if len(rsw.enabled_devices) == 0:
    #     raise ValueError("no devices connected")

    # rsw = RealsenseWrapper(rs_args, rs_args.rs_dev)
    # rsw.initialize()

    # rsw.flush_frames(rs_args.rs_fps * 5)
    # time.sleep(3)

    # device_sn = list(rsw.enabled_devices.keys())[0]
    # depth_scale = rsw.calib_data.get_data(device_sn)['depth']['depth_scale']
    # intr_mat = rsw.calib_data.get_data(device_sn)['depth']['intrinsic_mat']

    color_dict, depth_dict, trial_list = \
        get_filepaths_with_timestamps('/data/realsense')
    device_sn = list(color_dict[trial_list[0]].keys())[0]
    color_filepaths = [v[0] for _, v in
                       color_dict[trial_list[0]][device_sn].items()]
    depth_filepaths = [v[0] for _, v in
                       depth_dict[trial_list[0]][device_sn].items()]
    color_calibs = [v[1] for _, v in
                    color_dict[trial_list[0]][device_sn].items()]
    depth_calibs = [v[1] for _, v in
                    depth_dict[trial_list[0]][device_sn].items()]

    os.makedirs('/data/tmp/depth', exist_ok=True)
    for depth_filepath in depth_filepaths:
        output_filepath = depth_filepath.split('/')[-1].replace('npy', 'bin')
        output_filepath = os.path.join('/data/tmp/depth', output_filepath)
        depth_image = read_depth_file(depth_filepath)
        depth_image = depth_image[-848*480:]
        depth_image.astype(np.uint16).tofile(output_filepath)

    # 3. Setup extract and track classes ---------------------------------------
    PoseExtractor = OpenPosePoseExtractor(op_args)
    PoseTracker = Tracker(op_args, 30//(delay_switch+1))
    EST = ExtractSkeletonAndTrack(
        op_args, PoseExtractor, PoseTracker, enable_timer)
    EST.start()

    fig = plt.figure(figsize=(16, 8))
    pose, edge = None, None

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

            # # stand up
            # if c < 200:
            #     c += 1
            #     continue

            if c < 15:
                fps = {'PE': [], 'TK': []}

            # 5. grab image from realsense -------------------------------------
            # with Timer("update", enable_timer, False) as t:
            #     rsw.step(
            #         display=0,
            #         display_and_save_with_key=False,
            #         use_colorizer=True
            #     )
            # rs_time = t.duration
            # color_image = rsw.frames[device_sn]['color_framedata']  # h,w,c
            # depth_image = rsw.frames[device_sn]['depth_framedata']  # h,w
            # depth_colormap = rsw.frames[device_sn]['depth_color_framedata']
            # quit_key = visualize_rs(color_image=color_image,
            #                         depth_colormap=depth_colormap,
            #                         depth_scale=depth_scale,
            #                         winname=f'rs_{device_sn}')
            # if quit_key:
            #     break
            with open(color_calibs[c]) as f:
                calib = json.load(f)
            h_c = calib['color']['height']
            w_c = calib['color']['width']
            h_d = calib['depth']['height']
            w_d = calib['depth']['width']
            intr_mat = calib['depth']['intrinsic_mat']
            depth_scale = calib['depth']['depth_scale']
            color_image = read_color_file(color_filepaths[c],
                                          calib['color']['format'])
            color_image = color_image.reshape((h_c, w_c, -1))
            depth_image = read_depth_file(depth_filepaths[c])
            depth_image = depth_image[-h_d*w_d:].reshape(h_d, w_d)

            # 6. track without pose extraction ---------------------------------
            if delay_counter > 0:
                delay_counter -= 1
                EST.TK.no_measurement_predict_and_update()
                EST.PE.display(win_name=f'est_{device_sn}',
                               speed=display_speed,
                               scale=op_args.op_display,
                               image=None,
                               bounding_box=True,
                               tracks=EST.TK.tracks)
            else:
                delay_counter = delay_switch
                # 7. infer pose and track --------------------------------------
                EST.queue_input((color_image, None, None, None, False))
                est_out = EST.queue_output()
                (filered_skel, prep_time, infer_time, track_time, _) = est_out
                if not EST.PE.pyop.pose_empty:
                    EST.PE.pyop.convert_to_3d(
                        depth_image=depth_image,
                        intr_mat=intr_mat,
                        depth_scale=depth_scale
                    )
                    for track in EST.TK.tracks:
                        joints3d = EST.PE.pyop.pose_keypoints_3d[track.det_id]
                        joints3d = transform_3dpose(joints3d, depth_scale)
                        skelstorage.add(
                            track.track_id,
                            joints3d,
                            EST.PE.pyop.pose_keypoints[track.det_id][:, 2]
                        )
                skelstorage.check_valid_and_delete()
                status = EST.PE.display(win_name=f'est_{device_sn}',
                                        speed=display_speed,
                                        scale=op_args.op_display,
                                        image=None,
                                        bounding_box=False,
                                        tracks=EST.TK.tracks)
                if not status[0]:
                    break

            # 8. action recognition --------------------------------------------
            if time.time() - ar_start_time > int(ar_args.interval):
                with Timer("update", enable_timer, False) as t:
                    if len(skelstorage.skeletons) > 0:
                        # data  # M, 1, V, C
                        data, score = skelstorage.get_last_skel()
                        motion += [np.abs(data[score > 0.5]).sum()]
                        AR.append_data(
                            data[:op_args.op_max_true_body,
                                 :,
                                 :ar_args.num_joint,
                                 :],
                            score[:op_args.op_max_true_body,
                                  :,
                                  :ar_args.num_joint]
                        )
                        # logits = [#labels], prediction = from 0 - #labels-1
                        logits, prediction = AR.predict()
                        if len(skelstorage.skeletons) < 2:
                            logits = logits[:-3]
                            prediction = logits.index(max(logits))
                        logits = [round(i*100, 1) for i in logits]
                        # print(skelstorage.ids, logits, MAPPING[prediction+1])
                    recog_time = t.duration

            if recog_time is None:
                recog_time = 0.0

            # 8. printout ------------------------------------------------------
            if c % rs_args.rs_fps == 0:
                # printout(
                #     f"Step {c:12d} :: "
                #     f"{[i.get('color_timestamp', None) for i in rsw.frames.values()]} :: "  # noqa
                #     f"{[i.get('depth_timestamp', None) for i in rsw.frames.values()]}",  # noqa
                #     'i'
                # )
                # color_timestamp = list(rsw.frames.values())[0].get('color_timestamp')  # noqa
                color_timestamp = -1
                fps['PE'].append(1/infer_time)
                fps['TK'].append(1/track_time)
                printout(
                    f"Image : {color_timestamp} | "
                    f"#Skel filtered : {filered_skel} | "
                    f"#Tracks : {len(EST.TK.tracks)} | "
                    f"RS time : {rs_time:.3f} | "
                    f"Prep time : {prep_time:.3f} | "
                    f"Pose time : {infer_time:.3f} | "
                    f"Track time : {track_time:.3f} | "
                    f"Recog time : {recog_time:.3f} | "
                    f"FPS PE : {sum(fps['PE'])/len(fps['PE']):.3f} | "
                    f"FPS TK : {sum(fps['TK'])/len(fps['TK']):.3f}",
                    'i'
                )

            if len(skelstorage.skeletons) > 0:
                if len(skelstorage.skeletons[0]) > 0:
                    if len(skelstorage.skeletons[0]) <= ar_args.max_frame:
                        t_i = len(skelstorage.skeletons[0]) - 1
                    else:
                        t_i = ar_args.max_frame - 1
                    # data, score = skelstorage.get_last_skel()  # M, 1, V, C
                    data, mask = AR.DataProc.select_skeletons(ar_args.max_num_skeleton_true)  # M,T,V,C  # noqa
                    data = data[:, t_i:t_i+1, :, :]  # M, 1, V, C
                    data = data.transpose(3, 1, 2, 0)  # C, 1, V, M
                    if pose is None:
                        pose, edge, fig = visualize_3dskeleton_in_matplotlib(
                            data=np.expand_dims(data, axis=0),
                            graph='graph.openpose_b25_j15.Graph',
                            is_3d=True,
                            speed=1e-8,
                            text_per_t=[[c]+[MAPPING[prediction+1]]+logits],
                            fig=fig,
                            mode='openpose'
                        )
                    else:
                        visualize_3dskeleton_in_matplotlib_step(
                            data=np.expand_dims(data, axis=0),
                            t=0,
                            pose=pose,
                            edge=edge,
                            is_3d=True,
                            speed=1e-8,
                            text_per_t=[['FRAME:'] + [c] +
                                        [MAPPING[prediction+1]] +
                                        ['LOGITS:'] + logits],
                            fig=fig
                        )
                        jot = ["Head", "MSho",
                               "RSho", "RElb", "RHan",
                               "LSho", "LElb", "LHan",
                               "MHip", "RHip", "LHip",
                               "RKne", "RFee",
                               "LKne", "LFee",]
                        out = [f"{j} {i:.1f}" for j, i in zip(jot, EST.PE.pyop.pose_keypoints[0, :ar_args.num_joint, -1].tolist())]  # noqa
                        if len(motion) > 2:
                            print(out + [MAPPING[prediction+1]] + [round(sum([t - s for s, t in zip(motion, motion[1:])]), 1)])  # noqa

            # if not len(rsw.frames) > 0:
            #     printout(f"Empty...", 'w')
            #     continue

            c += 1
            if c > rs_args.rs_fps * rs_args.rs_steps or c > max_c:
                break

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        printout(f"{exc_type}, {fname}, {exc_tb.tb_lineno}", 'e')
        printout(f"Exception msg : {e}", 'e')
        traceback.print_tb(exc_tb)
        printout(f"Stopping RealSense devices...", 'i')
        # rsw.stop()

    finally:
        printout(f"Final RealSense devices...", 'i')
        # rsw.stop()

    EST.break_process_loops()

    printout(f"Finished...", 'i')
