import cv2
import json
import numpy as np
import os
import time
import traceback
import sys
from time import gmtime, strftime

from typing import Callable, Tuple, Optional

from rs_py.rs_run_devices import get_rs_parser
from rs_py.rs_run_devices import printout
from rs_py.rs_run_devices import RealsenseWrapper

from openpose.native.python.inference_rs import get_parser as get_op_parser
from openpose.native.python.inference_rs import OpenPosePoseExtractor
from openpose.native.python.inference_rs import Tracker
from openpose.native.python.inference_rs import ExtractSkeletonAndTrack
from openpose.native.python.utils import Timer

from infer.inference import ActionRecognition
from utils.parser import get_parser as get_ar_parser
from utils.parser import load_parser_args_from_config
from utils.utils import init_seed
from utils.visualization import visualize_3dskeleton_in_matplotlib
from utils.visualization import visualize_3dskeleton_in_matplotlib_step

FPS = 15
CAMERA_W = 848
CAMERA_H = 480

# HORITONTAL_BAR = np.ones((30, CAMERA_W, 3), dtype=np.uint8)*220
VERTICAL_BAR = np.concatenate(
    [np.ones((CAMERA_W, 5, 3), dtype=np.uint8)*np.array((255, 150, 0), dtype=np.uint8),  # noqa
     np.ones((CAMERA_W, 5, 3), dtype=np.uint8)*np.array((0, 0, 0), dtype=np.uint8),  # noqa
    #  np.ones((CAMERA_W, 6, 3), dtype=np.uint8)*np.array((255, 150, 0), dtype=np.uint8),  # noqa
    #  np.ones((CAMERA_W, 6, 3), dtype=np.uint8)*np.array((255, 255, 255), dtype=np.uint8),  # noqa
     np.ones((CAMERA_W, 5, 3), dtype=np.uint8)*np.array((255, 150, 0), dtype=np.uint8)],  # noqa
    axis=1
)

FIG, POSE, EDGE = None, None, None
MATPLOT = False

# # # cannot import this with opencv
# # # https://github.com/fourMs/MGT-python/issues/200
# import matplotlib.pyplot as plt
# FIG = plt.figure(figsize=(16, 8))
# MATPLOT = True

# OUTPUT_VIDEO = cv2.VideoWriter(
#     'project.avi',
#     cv2.VideoWriter_fourcc(*'DIVX'),
#     15,
#     (640, 480)
# )
OUTPUT_VIDEO = None


def get_args_rs():
    args, _ = get_rs_parser().parse_known_args()
    args.rs_steps = 1
    args.rs_fps = FPS
    args.rs_image_width = CAMERA_W
    args.rs_image_height = CAMERA_H
    args.rs_ir_emitter_power = 290
    args.rs_display_frame = 1
    args.rs_save_with_key = False
    args.rs_save_data = False
    args.rs_save_path = ''
    args.rs_postprocess = True  # max fps ~25 for 1280x720
    args.rs_vertical = True
    print("========================================")
    print(">>>>> args_rs <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")
    return args


def get_args_op():
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
    args.op_image_width = CAMERA_W
    args.op_image_height = CAMERA_H
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
    args.bytetracker_trackthresh = 0.2
    # args.bytetracker_trackbuffer = 30  # overwritten by op_track_buffer
    args.bytetracker_matchthresh = 0.9  # this is for (1-iou)
    args.bytetracker_mot20 = False
    args.bytetracker_fps = FPS
    print("========================================")
    print(">>>>> args_op <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")
    return args


def get_args_ar():
    init_seed(1)
    # label_mapping_file = "/data/07_AAGCN/model/ntu_15j/index_to_name.json"
    # label_mapping_file = "/data/07_AAGCN/model/ntu_15j_9l/index_to_name.json"
    # label_mapping_file = "/data/07_AAGCN/model/ntu_15j_5l/index_to_name.json"
    label_mapping_file = "/data/07_AAGCN/model/ntu_15j_4l/index_to_name.json"
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/weight/SGN-116-68208.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/weight/SGN-110-10670.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_5l_ntu_result/xview/aagcn_preprocess_sgn_model/230511130001/weight/SGN-120-8160.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_4l_ntu_result/xview/aagcn_preprocess_sgn_model/230512123001/weight/SGN-66-3168.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_4l_ntu_result/xsub/aagcn_preprocess_sgn_model/230512123001/weight/SGN-112-5824.pt"  # noqa
    weights = "/data/07_AAGCN/data/openpose_b25_j11_4l_ntu_result/xview/aagcn_preprocess_sgn_model/230517123001/weight/SGN-94-4512.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j11_4l_ntu_result/xsub/aagcn_preprocess_sgn_model/230517123001/weight/SGN-82-4264.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_5l_ntu_result/xview/aagcn_joint/230511130001/weight/Model-40-2720.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_ntu_result/xview/aagcn_joint/230414100001/weight/Model-50-29400.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_joint/230414100001/weight/Model-50-4850.pt"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_5l_ntu_result/xview/aagcn_preprocess_sgn_model/230511130001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_4l_ntu_result/xview/aagcn_preprocess_sgn_model/230512123001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_4l_ntu_result/xsub/aagcn_preprocess_sgn_model/230512123001/config.yaml"  # noqa
    config = "/data/07_AAGCN/data/openpose_b25_j11_4l_ntu_result/xview/aagcn_preprocess_sgn_model/230517123001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j11_4l_ntu_result/xsub/aagcn_preprocess_sgn_model/230517123001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_5l_ntu_result/xview/aagcn_joint/230511130001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_ntu_result/xview/aagcn_joint/230414100001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_joint/230414100001/config.yaml"  # noqa
    parser = get_ar_parser()
    parser.set_defaults(**{'config': config})
    args = load_parser_args_from_config(parser)
    args.weights = weights
    args.max_frame = 60
    args.max_num_skeleton_true = 2
    args.max_num_skeleton = 4
    args.num_joint = 11
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
    print(">>>>> args_ar <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")
    return args


def transform_3dpose(joints3d, depth_scale):
    # # 230511
    # matrix_z = np.array(
    #     [[0.99950274, -0.03090854, -0.00623903],
    #      [0.03090854,  0.92122211,  0.38780727],
    #      [-0.00623903, -0.38780727,  0.92171937]]
    # )
    # joints3d += np.array([0, -1500*depth_scale, 500*depth_scale])
    # # 230516
    # matrix_z = np.array(
    #     [[0.99953916, -0.02967045, -0.00641373],
    #      [0.02967045,  0.9102765,   0.41293627],
    #      [-0.00641373, -0.41293627, 0.91073734]]
    # )
    # joints3d += np.array([-700, -2200, 0])*depth_scale
    # # 230517
    # matrix_z = np.array(
    #     [[0.99657794, -0.08064464, -0.0181342],
    #      [0.08064464,  0.90048104,  0.42735271],
    #      [-0.0181342,  -0.42735271,  0.9039031]]
    # )
    # joints3d += np.array([-0, -1700, 1000])*depth_scale
    # joints3d = (matrix_z@joints3d.transpose()).transpose()
    # joints3d[:, 0] *= -1.
    # joints3d[:, 1] *= -0.7
    # # 230522
    # matrix_z = np.array(
    #     [[0.09203368, -0.94567705, -0.31180878],
    #      [0.94567705, -0.01504597,  0.32475919],
    #      [-0.31180878, -0.32475919,  0.89292035]]
    # )
    # joints3d += np.array([-1600, -1000, 1500])*depth_scale
    # joints3d = (matrix_z@joints3d.transpose()).transpose()
    # joints3d[:, 0] *= -1.
    # joints3d[:, 1] *= -1.  # to be ntu compat
    # # 230523
    # matrix_z = np.array(
    #     [[0.08971949, -0.9443577,  -0.31644739],
    #      [0.9443577,  -0.0202894,   0.32829389],
    #      [-0.31644739, -0.32829389,  0.88999111]]
    # )
    # 230531
    # matrix_z = np.array(
    #     [[0.09067461, -0.94602187, -0.31116032],
    #      [0.94602187, -0.01580074,  0.32371742],
    #      [-0.31116032, -0.32371742,  0.89352464]]
    # )
    # 23.06.06
    matrix_z = np.array(
        [[0.05289203, -0.97412245, -0.21974503],
         [0.97412245,  0.00190747,  0.22601284],
         [-0.21974503, -0.22601284,  0.94901545]]
    )
    joints3d += np.array([-2150, -700, 0])*depth_scale
    # joints3d += np.array([-1300, -800, 500])*depth_scale
    joints3d = (matrix_z@joints3d.transpose()).transpose()
    # joints3d[:, 0] *= -1.
    joints3d[:, 1] *= -1.  # to be ntu compat
    return joints3d


class Config():
    def __init__(self) -> None:
        # args
        self.args_rs = None
        self.args_op = None
        self.args_ar = None
        # Delay = predict and no update in tracker
        self.delay_switch = 2
        self.delay_counter = 2
        # For cv.imshow
        self.display_speed = 1
        # Runtime logging
        self.enable_timer = True
        self.fps = {'PE': [], 'TK': []}
        self.infer_time = -1
        self.track_time = -1
        self.filered_skel = -1
        self.prep_time = -1
        self.recog_time = -1
        self.rs_time = -1
        # Storage --------------------------------------------------------------
        # max length of data to store
        self.MAX_STORAGE_LEN = 100
        # buffer before removed
        self.BUFFER = 0
        # moving average and threshold for average
        self.MOVING_AVG = 9
        self.SCORE_THRES = 0.25
        # Motion and action filtering
        self.MOTION_DELAY = 10
        self.MOTION_COUNT = 20
        self.ACTION_COUNT = 10
        self.MAY_YS_DELAY = 6
        self.MAY_YS_COUNT = 9
        self.MAY_YL_DELAY = 18
        self.MAY_YL_COUNT = 21
        # TextParser -----------------------------------------------------------
        self.MOVE_THRES = 0.01
        # ActionHistory --------------------------------------------------------
        self.LOG_LEFT_OFFSET = 20
        self.LOG_TOP_OFFSET = 20
        self.LOG_HEIGHT = 14
        self.LOG_WIDTH = 30
        self.LOG_NUM_LB = 700//self.LOG_HEIGHT
        self.LOG_LB_OFFSET = 20
        self.BAR_LEFT_OFFSET = 0
        self.BAR_TOP_OFFSET = CAMERA_W - 40
        self.BAR_HEIGHT = 20
        self.BAR_WIDTH = 1
        self.BAR_NUM_LB = CAMERA_H*3//self.BAR_WIDTH
        self.BAR_LB_OFFSET = 20
        # Rule based action ----------------------------------------------------
        self.RULE_THRES_STAND = 0.05
        self.RULE_THRES_STAND_LONG = 0.10
        self.RULE_THRES_SIT = -0.05
        self.RULE_THRES_SIT_LONG = -0.09
        self.RULE_THRES_FALL = -0.20
        self.RULE_THRES_FALL_LONG = -0.40
        self.RULE_THRES_FALL_HEIGHT = 1.20

    @ property
    def valid_joints(self) -> list:
        # assert self.args_ar is not None
        if self.args_ar is not None:
            if self.args_ar.num_joint == 11:
                return [0, 1, 2, 3, 5, 6, 8, 9, 10, 12, 13]
            else:
                return [i for i in range(self.args_ar.num_joint)]
        else:
            return [0, 1, 2, 3, 5, 6, 8, 9, 10, 12, 13]


class Storage():
    def __init__(self, config: Config) -> None:
        self.ids = []  # idx
        self.raw_kypts = []  # idx, T, [V, 2]
        self.avg_kypts = []  # idx, T, [V, 2]
        self.raw_skels = []  # idx, T, [V, C]
        self.avg_skels = []  # idx, T, [V, C]
        self.raw_score = []  # idx, T, [V]
        self.avg_score = []  # idx, T, [V]
        # self.max__y = []  # idx, T
        # self.height = []  # idx, T
        # self.motion = []  # idx, [1] , experimental
        self.action = []  # T, [K] , experimental
        self.counter = []  # idx
        self.valid = []  # idx
        # max length of data to store
        self.max_len = config.MAX_STORAGE_LEN
        # buffer before removed
        self.buffer = config.BUFFER
        # moving average and threshold for average
        self.moving_avg = config.MOVING_AVG
        self.moving_avg_w = np.logspace(0, 1, self.moving_avg, base=2)
        self.score_thres = config.SCORE_THRES
        # Motion and action filtering
        self.motion_delay = config.MOTION_DELAY
        self.motion_count = config.MOTION_COUNT
        self.action_count = config.ACTION_COUNT
        self.max_ys_delay = config.MAY_YS_DELAY
        self.max_ys_count = config.MAY_YS_COUNT
        self.max_yl_delay = config.MAY_YL_DELAY
        self.max_yl_count = config.MAY_YL_COUNT

    def _clip_and_append(self, x_list, new_ele):
        return x_list[-self.max_len+1:] + [new_ele]

    def _avg(self, sid):
        # average of score
        valid_mask = np.array(self.raw_score[sid][-self.moving_avg:]) < self.score_thres  # t,v  # noqa
        valid_score = np.array(self.avg_score[sid][-self.moving_avg:])  # t,v
        masked_score = np.ma.array(valid_score, mask=valid_mask)
        if len(masked_score) < self.moving_avg:
            self.avg_score[sid][-1] = np.average(masked_score, axis=0).data  # noqa
        else:
            self.avg_score[sid][-1] = np.average(masked_score, axis=0, weights=self.moving_avg_w).data  # noqa
        # average of kpts and skels
        valid_kypts = np.array(self.avg_kypts[sid][-self.moving_avg:])  # t,v,c
        valid_skels = np.array(self.avg_skels[sid][-self.moving_avg:])  # t,v,c
        masked_kypts = np.ma.array(valid_kypts, mask=np.repeat(np.expand_dims(valid_mask, axis=-1), 2, axis=-1))  # noqa
        masked_skels = np.ma.array(valid_skels, mask=np.repeat(np.expand_dims(valid_mask, axis=-1), 3, axis=-1))  # noqa
        if len(masked_score) < self.moving_avg:
            self.avg_kypts[sid][-1] = np.average(masked_kypts, axis=0).data  # noqa
            self.avg_skels[sid][-1] = np.average(masked_skels, axis=0).data  # noqa
        else:
            self.avg_kypts[sid][-1] = np.average(masked_kypts, axis=0, weights=self.moving_avg_w).data  # noqa
            self.avg_skels[sid][-1] = np.average(masked_skels, axis=0, weights=self.moving_avg_w).data  # noqa

    def _diff_calc(self, x, s, c, d, mode=0):
        # y = mean(x[s>thres][-c:][:d] - x[s>thres][-c:][d:])
        default = 0.0
        if len(x) < 2:
            return default
        limit = len(x) - len(x)//2 if len(x) < c else d
        data_ = x[-c:]  # t, V, C
        valid = s[-c:] > self.score_thres  # t, V
        diff = []
        for m1, m2, v1, v2 in zip(data_[:limit], data_[limit:],
                                  valid[:limit], valid[limit:]):
            if (v1 & v2).sum() > 0:
                if mode == 0:
                    diff.append(np.nanmean(np.abs(m2[v1 & v2] -
                                                  m1[v1 & v2])))
                elif mode == 1:
                    diff.append(np.nanmean(np.max(m2[:, 1][v1 & v2]) -
                                           np.max(m1[:, 1][v1 & v2])))
        return default if len(diff) == 0 else np.nanmean(diff)

    def _motion_calc(self, skels, score):
        return self._diff_calc(
            skels, score, self.motion_count, self.motion_delay, 0)

    def _max_dys_calc(self, skels, score):
        return self._diff_calc(
            skels, score, self.max_ys_count, self.max_ys_delay, 1)

    def _max_dyl_calc(self, skels, score):
        return self._diff_calc(
            skels, score, self.max_yl_count, self.max_yl_delay, 1)

    def distance_between_2skel(self, x, s):
        # y = norm(x[0][s>thres] - x[1][s>thres])
        default = -1.0
        if len(x) < 2:
            return default
        else:
            x0, x1 = x[0], x[1]
            mask = (s[0] > self.score_thres) & (s[1] > self.score_thres)
            distance = np.linalg.norm(x1[mask] - x0[mask], axis=1)
            return np.mean(distance)

    def add(self, track_id, keypoints, skeletons, scores):
        if track_id in self.ids:
            sid = self.ids.index(track_id)
            self.raw_kypts[sid] = self._clip_and_append(self.raw_kypts[sid],
                                                        keypoints)
            self.raw_skels[sid] = self._clip_and_append(self.raw_skels[sid],
                                                        skeletons)
            self.raw_score[sid] = self._clip_and_append(self.raw_score[sid],
                                                        scores)
            self.avg_kypts[sid] = self._clip_and_append(self.avg_kypts[sid],
                                                        keypoints)
            self.avg_skels[sid] = self._clip_and_append(self.avg_skels[sid],
                                                        skeletons)
            self.avg_score[sid] = self._clip_and_append(self.avg_score[sid],
                                                        scores)
            self._avg(sid)
            self.valid[sid] = True
        else:
            self.ids.append(track_id)
            self.raw_kypts.append([keypoints])
            self.avg_kypts.append([keypoints])
            self.raw_skels.append([skeletons])
            self.avg_skels.append([skeletons])
            self.raw_score.append([scores])
            self.avg_score.append([scores])
            # self.max__y.append([skeletons[:, 1].max()])
            # self.height.append([skeletons[:, 1].max() -
            #                     skeletons[:, 1].min()])
            # self.motion.append(-1.0)
            self.counter.append(0)
            self.valid.append(True)

    def filter_action(self, logits, prediction):
        # append, clip, avg
        self.action.append(logits)
        if len(self.action) > self.action_count:
            self.action = self.action[-self.action_count:]
            avg = np.nanmean(self.action, axis=0)
            # self.action[-1] = avg
            return np.argmax(avg)
        else:
            return prediction

    def get_last_skel(self, valid_joints=None):
        raw_kypts, avg_kypts = [], []
        raw_skels, avg_skels = [], []
        raw_score, avg_score = [], []
        motion, max_dys, max_dyl = [], [], []
        M = len(self.avg_skels)
        for m in range(M):
            raw_kypts_m = self.raw_kypts[m]
            avg_kypts_m = self.avg_kypts[m]
            raw_skels_m = self.raw_skels[m]
            avg_skels_m = self.avg_skels[m]
            raw_score_m = self.raw_score[m]
            avg_score_m = self.avg_score[m]
            if valid_joints is not None:
                raw_kypts_m = [i[valid_joints, :] for i in raw_kypts_m]
                avg_kypts_m = [i[valid_joints, :] for i in avg_kypts_m]
                raw_skels_m = [i[valid_joints, :] for i in raw_skels_m]
                avg_skels_m = [i[valid_joints, :] for i in avg_skels_m]
                raw_score_m = [i[valid_joints] for i in raw_score_m]
                avg_score_m = [i[valid_joints] for i in avg_score_m]
            _skels = np.array(avg_skels_m)
            _score = np.array(raw_score_m)
            motion.append([self._motion_calc(_skels, _score)])
            max_dys.append([self._max_dys_calc(_skels, _score)])
            max_dyl.append([self._max_dyl_calc(_skels, _score)])
            raw_kypts.append([raw_kypts_m[-1]])
            avg_kypts.append([avg_kypts_m[-1]])
            raw_skels.append([raw_skels_m[-1]])
            avg_skels.append([avg_skels_m[-1]])
            raw_score.append([raw_score_m[-1]])
            avg_score.append([avg_score_m[-1]])
        motion = np.stack(motion, 0)
        max_dys = np.stack(max_dys, 0)
        max_dyl = np.stack(max_dyl, 0)
        raw_kypts = np.stack(raw_kypts, 0)
        avg_kypts = np.stack(avg_kypts, 0)
        raw_skels = np.stack(raw_skels, 0)
        avg_skels = np.stack(avg_skels, 0)
        raw_score = np.stack(raw_score, 0)
        avg_score = np.stack(avg_score, 0)
        # M, 1, V, C x2
        # M, 1, V, C x2
        # M, 1, V x2
        # M, 1 x2
        return (raw_kypts, avg_kypts, raw_skels, avg_skels,
                raw_score, avg_score, motion, max_dys, max_dyl)

    def delete(self, idx):
        self.ids.pop(idx)
        self.raw_kypts.pop(idx)
        self.avg_kypts.pop(idx)
        self.raw_skels.pop(idx)
        self.avg_skels.pop(idx)
        self.raw_score.pop(idx)
        self.avg_score.pop(idx)
        # self.motion.pop(idx)
        # self.max__y.pop(idx)
        # self.height.pop(idx)
        self.counter.pop(idx)
        self.valid.pop(idx)

    def check_valid_and_delete(self):
        num_data = len(self.counter)
        del_idx = []
        for i in range(num_data):
            if not self.valid[i]:
                self.counter[i] += 1
            if self.counter[i] > self.buffer:
                del_idx.append(i)
            self.valid[i] = False
        for i in del_idx:
            self.delete(i)

    def reset(self):
        num_data = len(self.counter)
        for idx in range(num_data):
            self.raw_kypts[idx] = self.raw_kypts[idx][-self.max_len//2:]
            self.avg_kypts[idx] = self.avg_kypts[idx][-self.max_len//2:]
            self.raw_skels[idx] = self.raw_skels[idx][-self.max_len//2:]
            self.avg_skels[idx] = self.avg_skels[idx][-self.max_len//2:]
            self.raw_score[idx] = self.raw_score[idx][-self.max_len//2:]
            self.avg_score[idx] = self.avg_score[idx][-self.max_len//2:]
            # self.motion[idx] = self.motion[idx][-self.motion_count:]
            # self.max__y[idx] = self.max__y[idx][-self.max_len//2:]
            # self.height[idx] = self.height[idx][-self.max_len//2:]


class TextParser():
    CM = {
        # "CYAN": (200, 1, 1),
        "CYAN": (200, 200, 1),
        # "CYAN": (255, 255, 1),
        "YELLOW": (1, 255, 255),
        # https://colorcodes.io/red/comic-book-red-color-codes/
        "RED": (36, 29, 237),
        "GREEN": (1, 255, 1),
        "PALEGREEN": (100, 255, 100),
        "SILVER": (192, 192, 192),
        "GRAY": (50, 50, 50),
        "WHITE": (255, 255, 255),
    }

    def __init__(self, config: Config, num_label=4, fps=15) -> None:
        self.MOVE_THRES = config.MOVE_THRES
        self.num_label = num_label
        self.fps = fps
        self.counter = [0 for _ in range(8)]
        self.last_text_id = -1

    def moving(self, movement):
        return True if movement > self.MOVE_THRES else False

    def text_parser(self, p_id, prediction, movement):
        p_id = ""
        text = ""
        if prediction < 0:
            text = f"Nobody is around."
            color = self.CM["GRAY"]
            text_id = 0
        elif prediction == 0:
            moving = self.moving(movement)
            if moving:
                text = f"Patient{p_id} is moving around."
                color = self.CM["YELLOW"]
                text_id = 1
                self.counter[1] += 1
            else:
                text = ""
                # text = f"Patient{p_id} is not doing anything."
                color = self.CM["SILVER"]
                text_id = 2
                self.counter[2] += 1
        else:
            # action shown for 3 s only.
            if self.counter[prediction+2] >= self.fps*5:
                return self.text_parser(p_id, prediction=0, movement=movement)
            else:
                if prediction == 1:
                    text = f"Patient{p_id} is sitting down. It is ok."
                    color = self.CM["GREEN"]
                    text_id = 3
                    self.counter[3] += 1
                elif prediction == 2:
                    text = f"Patient{p_id} is standing up. It is ok."
                    color = self.CM["GREEN"]
                    text_id = 4
                    self.counter[4] += 1
                elif prediction == 3:
                    text = f"Patient{p_id} is falling down. HELP!!!"
                    color = self.CM["RED"]
                    text_id = 5
                    self.counter[5] += 1
                elif prediction == 4:
                    text = f"Patient{p_id} is being attended to."
                    color = self.CM["YELLOW"]
                    text_id = 6
                elif prediction == 5:
                    text = f"Patient{p_id} is being checked."
                    color = self.CM["GREEN"]
                    text_id = 7
                # else:
                #     text = ""
                #     color = self.CM["GRAY"]
                #     text_id = 6

        # Reset count
        if text_id != self.last_text_id:
            self.counter[self.last_text_id] = 0

        self.last_text_id = text_id

        return text_id, text, color


class ActionHistory():
    def __init__(self, config: Config) -> None:
        self.LOG_LEFT_OFFSET = config.LOG_LEFT_OFFSET
        self.LOG_TOP_OFFSET = config.LOG_TOP_OFFSET
        self.LOG_HEIGHT = config.LOG_HEIGHT
        self.LOG_WIDTH = config.LOG_WIDTH
        self.LOG_NUM_LB = config.LOG_NUM_LB
        self.LOG_LB_OFFSET = config.LOG_LB_OFFSET
        self.BAR_LEFT_OFFSET = config.BAR_LEFT_OFFSET
        self.BAR_TOP_OFFSET = config.BAR_TOP_OFFSET
        self.BAR_HEIGHT = config.BAR_HEIGHT
        self.BAR_WIDTH = config.BAR_WIDTH
        self.BAR_NUM_LB = config.BAR_NUM_LB
        self.BAR_LB_OFFSET = config.BAR_LB_OFFSET
        self.log_label = []
        self.log_time_list = []
        self.log_color_list = []
        self.log_text_list = []
        self.log_image_memory = None
        self.bar_label = [0 for _ in range(CAMERA_H*3)]
        self.bar_color_list = [(0, 0, 0) for _ in range(CAMERA_H*3)]
        self._gmtime = gmtime()
        self.bar_time = [self._gmtime for _ in range(self.BAR_NUM_LB)]

    def time(self) -> None:
        self._gmtime = gmtime()

    def step(self, color: tuple, text: str, text_id: int) -> np.ndarray:

        if len(self.log_label) > 0:
            if text_id == self.log_label[-1]:
                return self.log_image_memory

        self.log_label = self.log_label[-self.LOG_NUM_LB:] + [text_id]
        self.log_time_list = self.log_time_list[-self.LOG_NUM_LB:] + \
            [f'{strftime("%y%m%d-%H%M%S", self._gmtime)} : ']
        self.log_color_list = self.log_color_list[-self.LOG_NUM_LB:] + [color]
        self.log_text_list = self.log_text_list[-self.LOG_NUM_LB:] + [text]

        image = np.zeros((CAMERA_W, CAMERA_H, 3), dtype=np.uint8)
        for idx, lab in enumerate(self.log_label):
            top = self.LOG_TOP_OFFSET+(self.LOG_HEIGHT*idx)
            btm = self.LOG_TOP_OFFSET+(self.LOG_HEIGHT*(idx+1))
            cv2.rectangle(image,
                          (self.LOG_LEFT_OFFSET, top),
                          (self.LOG_LEFT_OFFSET+self.LOG_WIDTH, btm),
                          self.log_color_list[idx], -1)
            _text = self.log_time_list[idx]
            _text += self.log_text_list[idx]
            image = cv2.putText(
                image,
                _text,
                (self.LOG_LEFT_OFFSET+self.LOG_WIDTH+self.LOG_LB_OFFSET, btm),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.log_color_list[idx],
                1,
                cv2.LINE_AA
            )
            # # ((32, 12), 5)
            # # ((26, 9), 4)
            # # ((20, 7), 3)
            # print(cv2.getTextSize(strftime("%y-%m-%d %H:%M:%S",
            #       gmtime()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1))
            # cv2.circle(rect, (10, 10), 20, (255, 255, 255), -1)

        # cv2.imshow("a", rect)
        # cv2.waitKey(100)

        self.log_image_memory = image
        return image

    def step_bar(self,
                 image: np.ndarray,
                 color: tuple,
                 text: str,
                 text_id: int) -> np.ndarray:

        self.bar_label = self.bar_label[-self.BAR_NUM_LB:] + [text_id]
        self.bar_color_list = self.bar_color_list[-self.BAR_NUM_LB:] + [color]
        self.bar_time = self.bar_time[-self.BAR_NUM_LB:] + [self._gmtime]
        image_bar = np.ones_like(image)
        for idx, lab in enumerate(self.bar_label):
            left = self.BAR_LEFT_OFFSET+(self.BAR_WIDTH*idx)
            right = self.BAR_LEFT_OFFSET+(self.BAR_WIDTH*(idx+1))
            cv2.rectangle(image_bar,
                          (left, self.BAR_TOP_OFFSET),
                          (right, self.BAR_TOP_OFFSET+self.BAR_HEIGHT),
                          self.bar_color_list[idx],
                          -1)
            if idx == 0:
                continue
            # if self.bar_time[idx].tm_min > self.bar_time[idx-1].tm_min:
            #     text = f'{self.bar_time[idx].tm_min}'
            if (self.bar_time[idx].tm_sec != self.bar_time[idx-1].tm_sec and
                    self.bar_time[idx].tm_sec % 10 == 0):
                text = f'{strftime("%y%m%d", self.bar_time[idx])}'
                image = cv2.putText(
                    image,
                    text,
                    (left+2, self.BAR_TOP_OFFSET-25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
                text = f'{strftime("%H%M%S", self.bar_time[idx])}'
                image = cv2.putText(
                    image,
                    text,
                    (left+2, self.BAR_TOP_OFFSET-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
                image = cv2.line(
                    image,
                    (left, self.BAR_TOP_OFFSET-35),
                    (left, self.BAR_TOP_OFFSET+self.BAR_HEIGHT),
                    (255, 255, 255),
                    2
                )

        mask = (image_bar-1).astype(bool)
        # image[mask] = cv2.addWeighted(image, 0.1, image_bar, 0.9, 0)[mask]
        image[mask] = image_bar[mask]
        return image


class App():
    def __init__(self) -> None:
        self.CF = None
        self.DS = None
        self.TP = None
        self.AH = None
        self.ET = None
        self.AR = None
        self.LABEL_MAPPING = None
        self.RS = None
        self.RS_INFO = {}

        self.valid_op_image = [0, None]

    def setup(self,
              config: Config,
              get_args_rs_func: Callable = get_args_rs,
              get_args_op_func: Callable = get_args_op,
              get_args_ar_func: Callable = get_args_ar
              ) -> None:
        # 1. args + setup ------------------------------------------------------
        # Overall configurations
        self.CF = config
        self.CF.args_rs = get_args_rs_func()
        self.CF.args_op = get_args_op_func()
        # self.CF.args_ar = get_args_ar_func()
        # For skeleton storage
        self.DS = Storage(self.CF)
        # For parsing sentences
        self.TP = TextParser(self.CF, fps=self.CF.args_rs.rs_fps)
        # Tracks previous action predictionsd
        self.AH = ActionHistory(self.CF)
        # 2. Setup extract and track classes -----------------------------------
        PoseExtractor = OpenPosePoseExtractor(self.CF.args_op)
        PoseTracker = Tracker(
            self.CF.args_op,
            self.CF.args_op.op_track_buffer//(self.CF.delay_switch+1)
        )
        self.ET = ExtractSkeletonAndTrack(self.CF.args_op,
                                          PoseExtractor,
                                          PoseTracker,
                                          self.CF.enable_timer)
        self.ET.start()
        # 3. setup aagcn -------------------------------------------------------
        # with open(self.CF.args_ar.label_mapping_file, 'r') as f:
        #     self.MAPPING = {int(i): j for i, j in json.load(f).items()}
        # self.AR = ActionRecognition(self.CF.args_ar)

    def setup_realsense(self):
        rsw = RealsenseWrapper(self.CF.args_rs, self.CF.args_rs.rs_dev)
        rsw.initialize_depth_sensor_ae()
        if len(rsw.enabled_devices) == 0:
            raise ValueError("no devices connected")
        rsw = RealsenseWrapper(self.CF.args_rs, self.CF.args_rs.rs_dev)
        rsw.initialize()
        rsw.flush_frames(self.CF.args_rs.rs_fps * 1)
        time.sleep(1)
        device_sn = list(rsw.enabled_devices.keys())[0]
        ds = rsw.calib_data.get_data(device_sn)['depth']['depth_scale']
        im = rsw.calib_data.get_data(device_sn)['depth']['intrinsic_mat']
        self.RS = rsw
        self.RS_INFO = {'device_sn': device_sn,
                        'depth_scale': ds,
                        'intr_mat': im, }

    def infer_pose_and_track(self, color_image, depth_image):
        skel_added = False
        # track without pose extraction ----------------------------------------
        if self.CF.delay_counter > 0:
            self.CF.delay_counter -= 1
            self.ET.TK.no_measurement_predict_and_update()
            for track in self.ET.TK.tracks:
                if track.is_activated:
                    # append the last raw skeleton
                    _sid = self.DS.ids.index(track.track_id)
                    self.DS.add(
                        track.track_id,
                        self.DS.raw_kypts[_sid][-1],
                        self.DS.raw_skels[_sid][-1],
                        self.DS.raw_score[_sid][-1]
                    )
                    skel_added = True
        else:
            self.CF.delay_counter = self.CF.delay_switch
            # infer pose and track ---------------------------------------------
            self.ET.queue_input((color_image, None, None, None, False))
            est_out = self.ET.queue_output()
            (self.CF.filered_skel, self.CF.prep_time,
                self.CF.infer_time, self.CF.track_time, _) = est_out
            if not self.ET.PE.pyop.pose_empty:
                self.ET.PE.pyop.convert_to_3d(
                    depth_image=depth_image,
                    intr_mat=self.RS_INFO['intr_mat'],
                    depth_scale=self.RS_INFO['depth_scale']
                )
                for track in self.ET.TK.tracks:
                    if track.is_activated:
                        joints3d = self.ET.PE.pyop.pose_keypoints_3d[track.det_id]  # noqa
                        joints3d = transform_3dpose(joints3d, self.RS_INFO['depth_scale'])  # noqa
                        self.DS.add(
                            track.track_id,
                            self.ET.PE.pyop.pose_keypoints[track.det_id][:, :2],
                            joints3d,
                            self.ET.PE.pyop.pose_keypoints[track.det_id][:, 2]
                        )
                        skel_added = True
            self.DS.check_valid_and_delete()
        status, op_image = self.ET.PE.display(
            win_name=f"est_{self.RS_INFO['device_sn']}",
            speed=self.CF.display_speed,
            scale=self.CF.args_op.op_display,
            image=None,
            bounding_box=True,
            tracks=self.ET.TK.tracks
        )
        return status, op_image, skel_added

    def infer_action(self,
                     c=0,
                     valid_joints: list | None = None,
                     custom_fn: Callable | None = None):
        # # data : M, 1, V, C
        # _, kpt, data, score = datastorage.get_last_skel()
        last_skel = self.DS.get_last_skel(valid_joints)
        (raw_kypts, avg_kypts,
         raw_skels, avg_skels,
         raw_score, avg_score,
         motion, max_dys, max_dyl) = last_skel
        distance = self.DS.distance_between_2skel(avg_skels, raw_score)
        distance_str = f"{f'{round(distance, 2):.2f}' if len(avg_skels) > 1 else ''}"  # noqa
        max_x = [round(i[0, :, 0].max(), 2) for i in avg_skels]
        max_y = [round(i[0, :, 1].max(), 2) for i in avg_skels]
        max_z = [round(i[0, :, 2].max(), 2) for i in avg_skels]
        min_z = [round(i[0, :, 2].min(), 2) for i in avg_skels]
        x_dif = [round(i[0, :, 0].max() - i[0, :, 0].min(), 2)
                 for i in avg_skels]
        y_dif = [round(i[0, :, 1].max() - i[0, :, 1].min(), 2)
                 for i in avg_skels]
        ratio = [round(j / i, 2) for i, j in zip(x_dif, y_dif)]

        # # DL MODEL ------
        # self.AR.append_data(avg_skels, avg_score)
        # logits, prediction = self.AR.predict()
        # logits, prediction = self.DS.filter_action(
        #     logits, prediction)

        if custom_fn is not None:
            custom_output = custom_fn(last_skel)
        # num_joints = avg_skels.shape[2]
        # if num_joints == 11:
        #     head = avg_skels[:, 0, 0]
        #     hand_l = avg_skels[:, 0, 3]
        #     hand_r = avg_skels[:, 0, 5]
        # elif num_joints == 15:
        #     head = avg_skels[:, 0, 0]
        #     hand_l = avg_skels[:, 0, 4]
        #     hand_r = avg_skels[:, 0, 7]
        # else:
        #     raise ValueError("Unknown number of joints...")

        # RULE BAESD -----
        logits = [0]
        prediction = 0
        # 1. Falling, ys < RULE_THRES_FALL_HEIGHT (+ve)
        if max_y[0] < self.CF.RULE_THRES_FALL_HEIGHT:
            prediction = 3
        # 2a. Falling, d_ys < RULE_THRES_FALL (-ve)
        elif max_dys[0] < self.CF.RULE_THRES_FALL:
            prediction = 3
        # 2b. Falling, d_ys < RULE_THRES_FALL_LONG (-ve)
        elif max_dyl[0] < self.CF.RULE_THRES_FALL_LONG:
            prediction = 3
        # 3a. Sit, d_ys < RULE_THRES_SIT (-ve)
        elif max_dys[0] < self.CF.RULE_THRES_SIT:
            # 5.1. (Slowly) Falling, ys < RULE_THRES_FALL_HEIGHT (+ve)
            if max_y[0] < self.CF.RULE_THRES_FALL_HEIGHT:
                prediction = 3
            # 5.2. Sitting
            else:
                prediction = 1
        # 3b. Sit, d_yl < RULE_THRES_SIT_LONG (-ve)
        elif max_dyl[0] < self.CF.RULE_THRES_SIT_LONG:
            # 4.1. (Slowly) Falling, ys < RULE_THRES_FALL_HEIGHT (+ve)
            if max_y[0] < self.CF.RULE_THRES_FALL_HEIGHT:
                prediction = 3
            # 4.2. Sitting
            else:
                prediction = 1
        # 4a. Standing, d_ys > RULE_THRES_STAND (+ve)
        elif max_dys[0] > self.CF.RULE_THRES_STAND:
            prediction = 2
        # 4b. Standing, d_yl > RULE_THRES_STAND_LONG (+ve)
        elif max_dyl[0] > self.CF.RULE_THRES_STAND_LONG:
            prediction = 2
        # 5. two persons in the scene
        if len(self.DS.avg_skels) > 1:
            if len(self.DS.avg_skels[0]) > 5 and len(self.DS.avg_skels[1]) > 5:
                if distance < 1.3:
                    prediction = 4
                else:
                    prediction = 5
        logits = np.zeros(6)
        logits[prediction] = 1.0
        prediction = self.DS.filter_action(logits, prediction)
        printout(
            f'{c:04} | '
            # "#Pose > 30% :", round((score > 0.3).sum(), 1), "|",
            f"ID : {self.DS.ids} | "
            f"#Skels : {[len(s) for s in self.DS.avg_skels]} | "
            f"{[f'{round(a, 2):.2f}' for a in self.DS.action[-1]]} | "
            f"Distance : {distance_str}"  # noqa
            # [f'{round(i, 2):.2f}' for i in np.linalg.norm(avg_skels[0] - avg_skels[1], axis=1).tolist()]  # noqa
            # "EAT :",
            # [f'{round(i, 2):.2f}' for i in np.linalg.norm(hand_l - head, axis=1).tolist()],  # noqa
            # [f'{round(i, 2):.2f}' for i in np.linalg.norm(hand_r - head, axis=1).tolist()]  # noqa
           f"", 'i'
        )
        printout(
            # "Logits :", [round(i*100, 1) for i in logits], "|",
            # "Action :", self.LABEL_MAPPING[prediction+1][:10], "|",
            # "MaxX :", max_x, "|",
            f"MaxY : {[f'{i:.2f}' for i in max_y]} | "
            f"MaxYsD : {[f'{round(i[0], 2):.2f}' for i in max_dys]} | "
            f"MaxYlD : {[f'{round(i[0], 2):.2f}' for i in max_dyl]} | "
            f"MaxZ : {[f'{i:.2f}' for i in max_z]} | "
            f"MinZ : {[f'{i:.2f}' for i in min_z]} | "
            f"H : {[f'{i:.2f}' for i in y_dif]} | "
            f"[H/W] : {[f'{i:.2f}' for i in ratio]} | "
            # ["MOVING" if m[0] > 0.015 else "STILL"
            # for m in motion]
            f"M : {[f'{round(m[0], 3):.3f}' for m in motion]}"
            f"", 'i'
        )

        return raw_kypts, raw_score, avg_skels, motion, prediction, distance_str

    @staticmethod
    def visualize_3dplot(data, alpha, num_joint):
        # data: C, 1, V, M
        if POSE is None:
            if num_joint == 11:
                graph = 'graph.openpose_b25_j11.Graph'
            else:
                graph = 'graph.openpose_b25_j15.Graph'
            POSE, EDGE, FIG = \
                visualize_3dskeleton_in_matplotlib(
                    data=np.expand_dims(data, axis=0),
                    graph=graph,
                    is_3d=True,
                    speed=1e-8,
                    fig=FIG,
                    mode='openpose',
                    alpha=alpha
                )
        else:
            visualize_3dskeleton_in_matplotlib_step(
                data=np.expand_dims(data, axis=0),
                t=0,
                pose=POSE,
                edge=EDGE,
                is_3d=True,
                speed=1e-8,
                fig=FIG,
                alpha=alpha
            )
            # jot = ["Head", "MSho",
            #        "RSho", "RElb", "RHan",
            #        "LSho", "LElb", "LHan",
            #        "MHip", "RHip", "LHip",
            #        "RKne", "RFee",
            #        "LKne", "LFee",]
            # out = [f"{j} {i:.1f}" for j, i in zip(jot, EST.PE.pyop.pose_keypoints[0, valid_joints, -1].tolist())]  # noqa
            # print(out)

    # UNUSED ---
    @staticmethod
    def visualize_rs(color_image: np.ndarray,
                     depth_colormap: np.ndarray,
                     depth_scale: float,
                     winname: str):
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

    def visualize_output(self,
                         color_image: np.ndarray,
                         depth_image: np.ndarray,
                         op_image: Optional[np.ndarray] = None,
                         kpt: Optional[np.ndarray] = None,
                         prediction: int = -1,
                         motion: np.ndarray | float = 0.,
                         distance_str: str = '',
                         vertical: bool = False):
        depth_image = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.06),
            cv2.COLORMAP_BONE
        )
        if op_image is not None and op_image.size > 0 and kpt is not None:
            op_image[op_image.sum(-1) ==
                     0] = depth_image[op_image.sum(-1) == 0]
            self.valid_op_image[0] = 0
            self.valid_op_image[1] = op_image
            text_id, text, color = self.TP.text_parser(
                self.DS.ids[0], prediction, motion[0, 0])
        else:
            if (self.valid_op_image[1] is not None and
                    self.valid_op_image[0] < self.CF.delay_counter):
                self.valid_op_image[0] += 1
                op_image = self.valid_op_image[1]
            else:
                op_image = depth_image
            text_id, text, color = self.TP.text_parser(-1, -1, -1)
        if vertical:
            op_image = np.rot90(op_image, -1).copy()
        printout(text, 'i')
        # 1. Left section
        if vertical:
            color_image = np.rot90(color_image, -1).copy()
        crop = color_image[:50, :]
        black_rect = np.zeros(crop.shape, dtype=np.uint8)
        overlay = cv2.addWeighted(crop, 0.3, black_rect, 0.7, 1.0)
        color_image[:50, :] = overlay
        if text is not None:
            color_image = cv2.putText(color_image,
                                      text,
                                      (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.8,
                                      color,
                                      1,
                                      cv2.LINE_AA)
        # if kpt is not None:
        #     pos = kpt.astype(int)[:, 0, 0, :]  # [M, 2]
        #     if vertical:
        #         pos = np.flip(pos)
        #         pos[:, 0] = CAMERA_H - pos[:, 0]
        #     for pos_i in pos:
        #         cv2.circle(color_image, pos_i, 10, (255, 255, 255), -1)
        # 2. Middle section
        if vertical:
            depth_image = np.rot90(depth_image, -1).copy()
        # 3. Right section
        self.AH.time()
        history = self.AH.step(color, text, text_id)
        image = np.hstack(
            [color_image, VERTICAL_BAR,
             op_image, VERTICAL_BAR,
             history]
        )
        # 4. Bottom running bar
        image = self.AH.step_bar(image, color, text, text_id)

        cv2.putText(image,
                    "Distance : " + distance_str,
                    (10, CAMERA_W-10, ),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    (0, 0, 200),
                    2,
                    cv2.LINE_AA)
        return image


if __name__ == "__main__":

    config = Config()
    app = App()
    app.setup(config)

    cv2.namedWindow("Monitor", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Monitor",
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Setup realsense ----------------------------------------------------------
    app.setup_realsense()

    # MAIN LOOP ----------------------------------------------------------------
    ar_start_time = time.time()

    # try:
    c = 0
    max_c = int(1e8)

    while True:

        if c < 15:
            fps = {'PE': [], 'TK': []}

        # 1. grab image from realsense -------------------------------------
        with Timer("rsw", app.CF.enable_timer, False) as t:
            app.RS.step(
                display=0,
                display_and_save_with_key=False,
                use_colorizer=False
            )
        app.CF.rs_time = t.duration
        frame = app.RS.frames[app.RS_INFO['device_sn']]
        color_image = frame['color_framedata']  # h,w,c
        depth_image = frame['depth_framedata']  # h,w
        if app.CF.args_rs.rs_postprocess:
            depth_image = cv2.resize(depth_image, (color_image.shape[1],
                                                   color_image.shape[0]))
        # Remove the border pixels as measurements there are not good
        color_image[:20, :, :] = 169  # right border
        color_image[-20:, :, :] = 169  # left border
        color_image[depth_image > 2800] = 169
        color_image[depth_image < 300] = 169
        # depth_colormap = rsw.frames[device_sn]['depth_color_framedata']
        # quit_key = app.visualize_rs(color_image=color_image,
        #                             depth_colormap=depth_colormap,
        #                             depth_scale=depth_scale,
        #                             winname=f'rs_{device_sn}')
        # if quit_key:
        #     break

        # 2. track without pose extraction ---------------------------------
        # 3. infer pose and track ------------------------------------------
        status, op_image, skel_added = app.infer_pose_and_track(
            color_image, depth_image)

        # try:
        #     status, op_image, skel_added = app.infer_pose_and_track(
        #         color_image, depth_image)
        # except Exception as e:
        #     printout(f"{e.message}", 'w')
        #     printout(f"{e.args}", 'w')
        #     continue

        if not status:
            break

        # 4. action recognition --------------------------------------------
        # if time.time() - ar_start_time < int(app.CF.args_ar.interval):
        #     continue

        distance_str = ""
        raw_kypts = None
        prediction = -1
        motion = 0.
        with Timer("AR", app.CF.enable_timer, False) as t:
            # if len(app.DS.skeletons) > 0:
            if skel_added:
                (raw_kypts, raw_score, avg_skels,
                    motion, prediction, distance_str) = app.infer_action(
                        c,
                        app.CF.valid_joints
                )
            else:
                printout(f'{c:04} No skeletons... '
                         f'{len(app.DS.avg_skels)} '
                         f'{app.ET.PE.pyop.pose_scores}',
                         'i')

        app.CF.recog_time = t.duration
        if app.CF.recog_time is None:
            app.CF.recog_time = -1

        # Visualize results -----------
        if not MATPLOT:
            depth_image[depth_image > 2800] = 2800
            depth_image[depth_image < 300] = 300
            image = app.visualize_output(
                color_image,
                depth_image,
                op_image,
                raw_kypts,
                prediction,
                motion,
                distance_str,
                app.CF.args_rs.rs_vertical
            )
            # Face sensor
            if raw_kypts is not None:
                pos = raw_kypts.astype(int)[:, 0, 0, :]  # [M, 2]
                if app.CF.args_rs.rs_vertical:
                    pos = np.flip(pos)
                    pos[:, 0] = CAMERA_H - pos[:, 0]
                for pos_i in pos:
                    cv2.circle(color_image, pos_i, 10, (255, 255, 255), -1)

            # Unblur blur due to resize to fullscreen:
            # https://stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencv
            image = cv2.resize(image, (1920, 1080))
            blurr = cv2.GaussianBlur(image, (0, 0), 3)
            image = cv2.addWeighted(image, 1.5, blurr, -0.5, 0)

            if OUTPUT_VIDEO is not None:
                OUTPUT_VIDEO.write(image)
            else:
                cv2.imshow("Monitor", image)
                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    break

        if MATPLOT:
            # M, 1, V, C -> C, 1, V, M
            avg_skels = avg_skels.transpose(3, 1, 2, 0)[:, :, :, 0:1]  # noqa
            avg_skels[1] *= -1.0
            alpha = raw_score[0, 0].tolist()
            num_joint = app.CF.args_ar.num_joint
            app.visualize_3dplot(avg_skels, alpha, num_joint)

        # 8. printout ------------------------------------------------------
        # if c % app.CF.args_rs.rs_fps == 0:
        #     # printout(
        #     #     f"Step {c:12d} :: "
        #     #     f"{[i.get('color_timestamp', None) for i in app.RS.frames.values()]} :: "  # noqa
        #     #     f"{[i.get('depth_timestamp', None) for i in app.RS.frames.values()]}",  # noqa
        #     #     'i'
        #     # )
        #     color_timestamp = list(app.RS.frames.values())[0].get('color_timestamp')  # noqa
        #     fps['PE'].append(1/app.CF.infer_time)
        #     fps['TK'].append(1/app.CF.track_time)
        #     printout(
        #         f"Image : {color_timestamp} | "
        #         f"#Skel filtered : {app.CF.filered_skel} | "
        #         f"#Tracks : {len(app.ET.TK.tracks)} | "
        #         f"RS time : {app.CF.rs_time:.3f} | "
        #         f"Prep time : {app.CF.prep_time:.3f} | "
        #         f"Pose time : {app.CF.infer_time:.3f} | "
        #         f"Track time : {app.CF.track_time:.3f} | "
        #         f"Recog time : {app.CF.recog_time:.3f} | "
        #         f"FPS PE : {sum(fps['PE'])/len(fps['PE']):.3f} | "
        #         f"FPS TK : {sum(fps['TK'])/len(fps['TK']):.3f}",
        #         'i'
        #     )

        if not len(app.RS.frames) > 0:
            printout(f"Empty...", 'w')
            continue

        if c % (app.CF.args_rs.rs_fps*60) == 0 and c > 0:
            app.DS.reset()

        c += 1

    # except Exception as e:
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     printout(f"{exc_type}, {fname}, {exc_tb.tb_lineno}", 'e')
    #     printout(f"Exception msg : {e}", 'e')
    #     traceback.print_tb(exc_tb)
    #     printout(f"Stopping RealSense devices...", 'i')
    #     app.RS.stop()

    # finally:
    #     printout(f"Final RealSense devices...", 'i')
    #     app.RS.stop()

    app.RS.stop()

    if OUTPUT_VIDEO is not None:
        OUTPUT_VIDEO.release()
    app.ET.break_process_loops()
    printout(f"Finished...", 'i')
