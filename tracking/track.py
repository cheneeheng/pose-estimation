import argparse
import numpy as np

from typing import Optional

from .utils_track import create_detections

from submodules.ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from submodules.ByteTrack.yolox.tracker.byte_tracker import STrack
from submodules.ByteTrack.yolox.tracker.byte_tracker import joint_stracks

from submodules.OC_SORT.trackers.ocsort_tracker.ocsort import OCSort

from submodules.StrongSORT.opts import opt
from submodules.StrongSORT.deep_sort.tracker import Tracker as StrongSortTracker
from submodules.StrongSORT.deep_sort.nn_matching import NearestNeighborDistanceMetric  # noqa


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class DeepSortTrackerArgs:
    def __init__(self, args) -> None:
        self.metric = args.deepsort_metric
        self.opt = opt
        self.opt.NSA = args.deepsort_opt_nsa
        self.opt.EMA = args.deepsort_opt_ema
        self.opt.MC = args.deepsort_opt_mc
        self.opt.woC = args.deepsort_opt_woc
        self.opt.EMA_alpha = args.deepsort_opt_emaalpha
        self.opt.MC_lambda = args.deepsort_opt_mclambda
        self.opt.max_cosine_distance = args.deepsort_opt_maxcosinedistance
        self.opt.nn_budget = args.deepsort_opt_nnbudget


class ByteTrackerArgs:
    def __init__(self, args) -> None:
        # tracking confidence threshold, splits detection to high and low
        # detection confidence groups.
        # track_thresh = 0.5
        self.track_thresh = args.bytetracker_trackthresh
        # the frames for keep lost tracks
        self.track_buffer = args.bytetracker_trackbuffer
        # matching threshold for tracking (IOU)
        self.match_thresh = args.bytetracker_matchthresh
        # match_thresh = 0.3
        # test mot20 dataset
        self.mot20 = args.bytetracker_mot20


class OCSortArgs:
    def __init__(self, args) -> None:
        self.det_thresh = args.ocsort_detthresh
        self.max_age = args.ocsort_maxage
        self.min_hits = args.ocsort_minhits
        self.iou_threshold = args.ocsort_iouthreshold
        self.delta_t = args.ocsort_deltat
        # ASSO_FUNCS in ocsort.py
        self.asso_func = args.ocsort_assofunc
        # momentum value
        self.inertia = args.ocsort_inertia
        self.use_byte = args.ocsort_usebyte


class StrongSortTrackerArgs:
    def __init__(self, args) -> None:
        self.metric = args.strongsort_metric
        # # DEFAULTS
        # self.bot = True  # the REID model used
        # if self.bot:
        #     self.matching_threshold = 0.4
        # else:
        #     self.matching_threshold = 0.3
        # if self.mc_lambda is not None:
        #     self.matching_threshold += 0.05
        # if self.ema_alpha is not None:
        #     self.nn_budget = 1
        # else:
        #     self.nn_budget = 100
        self.opt = opt
        self.opt.NSA = args.strongsort_opt_nsa
        self.opt.EMA = args.strongsort_opt_ema
        self.opt.EMA_alpha = args.strongsort_opt_emaalpha
        self.opt.MC = args.strongsort_opt_mc
        self.opt.MC_lambda = args.strongsort_opt_mclambda
        self.opt.woC = args.strongsort_opt_woc


# Tracking inspired by : https://github.com/ortegatron/liveposetracker
class Tracker():

    def __init__(self,
                 input_args: argparse.Namespace,
                 max_age: int = 30) -> None:
        """Intializes the tracker class.

        Args:
            max_age (int, optional): How long an untracked obj stays alive.
                Same as buffer_size in byte tracker. Defaults to 30.
        """
        self.detections = None
        if input_args.op_track_deepsort:
            self.name = 'deep_sort'
            args = DeepSortTrackerArgs(input_args)
            metric = NearestNeighborDistanceMetric(
                args.metric, args.opt.max_cosine_distance, args.opt.nn_budget)
            self.tracker = StrongSortTracker(
                metric, max_age=max_age, options=args.opt)
        elif input_args.op_track_bytetrack:
            self.name = 'byte_tracker'
            args = ByteTrackerArgs(input_args)
            args.track_buffer = max_age
            self.tracker = BYTETracker(args)
        elif input_args.op_track_ocsort:
            self.name = 'oc_sort'
            args = OCSortArgs(input_args)
            args.max_age = max_age
            self.tracker = OCSort(**args.__dict__)
        elif input_args.op_track_strongsort:
            self.name = 'strong_sort'
            args = StrongSortTrackerArgs(input_args)
            metric = NearestNeighborDistanceMetric(
                args.metric, args.opt.max_cosine_distance, args.opt.nn_budget)
            self.tracker = StrongSortTracker(
                metric, max_age=max_age, options=args.opt)
        else:
            raise ValueError("Not implemented...")

    @property
    def tracks(self):
        if self.name == 'deep_sort' or self.name == 'strong_sort':
            return self.tracker.tracks
        elif self.name == 'byte_tracker':
            return self.tracker.tracked_stracks
        elif self.name == 'oc_sort':
            return self.tracker.tracks

    def predict(self):
        if self.name == 'byte_tracker':
            raise ValueError("Not used...")
        elif self.name == 'oc_sort':
            raise ValueError("Not used...")
        self.tracker.predict()

    def update(self,
               boxes: np.ndarray,
               scores: np.ndarray,
               keypoints: np.ndarray,
               heatmaps: np.ndarray,
               image_size: Optional[tuple] = None):
        if self.name == 'deep_sort' or self.name == 'strong_sort':
            self.predict()
            self.detections = self._create_detections(
                boxes, scores, keypoints, heatmaps, image_size)
            self.tracker.update(self.detections)
        elif self.name == 'byte_tracker' or self.name == 'oc_sort':
            self.detections = self._create_detections_np(boxes, scores)
            self.tracker.update(self.detections)

    @staticmethod
    def _create_detections(boxes: np.ndarray,
                           scores: np.ndarray,
                           keypoints: np.ndarray,
                           heatmaps: np.ndarray,
                           image_size: tuple):
        s_h = heatmaps.shape[0]/image_size[1]
        s_w = heatmaps.shape[1]/image_size[0]
        return create_detections(
            keypoints, scores, boxes, heatmaps, [s_w, s_h])

    @staticmethod
    def _create_detections_np(boxes: np.ndarray,
                              scores: np.ndarray,):
        scores = np.expand_dims(scores, 1)
        return np.concatenate([boxes, scores], axis=1)

    def no_measurement_predict_and_update(self):
        if self.name == 'deep_sort' or self.name == 'strong_sort':
            for tracks in self.tracker.tracks:
                tracks.mean, tracks.covariance = \
                    self.tracker.kf.predict(
                        tracks.mean,
                        tracks.covariance,
                    )
                tracks.mean, tracks.covariance = \
                    self.tracker.kf.update(
                        tracks.mean,
                        tracks.covariance,
                        tracks.mean[:4]
                    )
        elif self.name == 'byte_tracker':
            strack_pool = joint_stracks(self.tracker.tracked_stracks,
                                        self.tracker.lost_stracks)
            # inplace update
            STrack.multi_predict(strack_pool)
        elif self.name == 'oc_sort':
            for tracker in self.tracker.trackers:
                _bbox = tracker.predict()
                tracker.update(_bbox)
