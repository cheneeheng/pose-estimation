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


class DeepSortTrackerArgs:
    def __init__(self) -> None:
        self.metric = 'cosine'

        self.opt = opt
        self.opt.NSA = False
        self.opt.EMA = None
        self.opt.MC = None
        self.opt.woC = False
        self.opt.EMA_alpha = None
        self.opt.MC_lambda = None

        self.opt.max_cosine_distance = 0.2
        self.opt.nn_budget = 100


class ByteTrackerArgs:
    # tracking confidence threshold, splits detection to high and low
    # detection confidence groups.
    track_thresh = 0.5
    # the frames for keep lost tracks
    track_buffer = 30
    # matching threshold for tracking (IOU)
    match_thresh = 0.8
    # test mot20 dataset
    mot20 = False


class OCSortArgs:
    def __init__(self) -> None:
        self.det_thresh = 0.5
        self.max_age = 30
        self.min_hits = 3
        self.iou_threshold = 0.3
        self.delta_t = 3
        # ASSO_FUNCS in ocsort.py
        self.asso_func = "iou"
        # momentum value
        self.inertia = 0.2
        self.use_byte = False


class StrongSortTrackerArgs:
    def __init__(self) -> None:
        self.metric = 'cosine'

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

        nsa = True
        # nsa = False

        woc = True
        # woc = False

        ema_alpha = 0.9
        # ema_alpha = None

        mc_lambda = 0.98
        # mc_lambda = None

        self.opt = opt
        self.opt.NSA = nsa
        self.opt.EMA = True if ema_alpha is not None else False
        self.opt.MC = True if mc_lambda is not None else False
        self.opt.woC = woc
        self.opt.EMA_alpha = ema_alpha
        self.opt.MC_lambda = mc_lambda


# Tracking inspired by : https://github.com/ortegatron/liveposetracker
class Tracker():

    def __init__(self, mode: str = 'deep_sort', max_age: int = 30) -> None:
        """Intializes the tracker class.

        Args:
            max_age (int, optional): How long an untracked obj stays alive.
                Same as buffer_size in byte tracker. Defaults to 30.
        """
        self.detections = None

        if mode == 'deep_sort':
            self.name = 'deep_sort'
            args = DeepSortTrackerArgs()
            metric = NearestNeighborDistanceMetric(
                args.metric, args.opt.max_cosine_distance, args.opt.nn_budget)
            self.tracker = StrongSortTracker(
                metric, max_age=max_age, options=args.opt)
        elif mode == 'byte_tracker':
            self.name = 'byte_tracker'
            args = ByteTrackerArgs()
            args.track_buffer = max_age
            self.tracker = BYTETracker(args)
        elif mode == 'oc_sort':
            self.name = 'oc_sort'
            args = OCSortArgs()
            args.max_age = max_age
            self.tracker = OCSort(**args.__dict__)
        elif mode == 'strong_sort':
            self.name = 'strong_sort'
            args = StrongSortTrackerArgs()
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
