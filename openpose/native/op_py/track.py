import numpy as np

from .skeleton import PyOpenPoseNative
from .utils import Timer
from .utils_track import create_detections

from submodules.deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from submodules.deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric  # noqa

from submodules.ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from submodules.ByteTrack.yolox.tracker.byte_tracker import STrack
from submodules.ByteTrack.yolox.tracker.byte_tracker import joint_stracks

from submodules.OC_SORT.trackers.ocsort_tracker.ocsort import OCSort


class DeepSortTrackerArgs():
    metric = 'cosine'
    # max_cosine_distance = 0.2
    matching_threshold = 0.2
    nn_budget = None


class ByteTrackerArgs():
    # tracking confidence threshold, splits detection to high and low
    # detection confidence groups.
    track_thresh = 0.5
    # the frames for keep lost tracks
    track_buffer = 30
    # matching threshold for tracking (IOU)
    match_thresh = 0.8
    # test mot20 dataset
    mot20 = False


class OCSortArgs():
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


# Tracking inspired by : https://github.com/ortegatron/liveposetracker
class Tracker():

    def __init__(self, mode: str = 'deep_sort', max_age: int = 30) -> None:
        """Intializes the tracker class.

        Args:
            max_age (int, optional): How long an untracked obj stays alive.
                Same as buffer_size in byte tracker. Defaults to 30.
        """
        if mode == 'deep_sort':
            args = DeepSortTrackerArgs()
            metric = NearestNeighborDistanceMetric(
                args.metric, args.matching_threshold, args. nn_budget)
            self.tracker = DeepSortTracker(metric, max_age=max_age)
            self.name = 'deep_sort'
        elif mode == 'byte_tracker':
            args = ByteTrackerArgs()
            args.track_buffer = max_age
            self.tracker = BYTETracker(args)
            self.name = 'byte_tracker'
        elif mode == 'oc_sort':
            args = OCSortArgs()
            args.max_age = max_age
            self.tracker = OCSort(**args.__dict__)
            self.name = 'oc_sort'
        else:
            raise ValueError("Not implemented...")
        self.detections = None

    @property
    def tracks(self):
        if self.name == 'deep_sort':
            return self.tracker.tracks
        elif self.name == 'byte_tracker':
            return self.tracker.tracked_stracks
        elif self.name == 'oc_sort':
            return self.tracker.tracks

    def predict(self):
        if self.name == 'byte_tracker':
            raise ValueError("Not used...")
        self.tracker.predict()

    def update(self, pyop: PyOpenPoseNative, image_size: tuple = None):
        if self.name == 'deep_sort':
            self.detections = self._create_detections(pyop, image_size)
            self.tracker.update(self.detections)
        elif self.name == 'byte_tracker':
            self.detections = self._create_detections(pyop)
            self.tracker.update(self.detections)
        elif self.name == 'oc_sort':
            self.detections = self._create_detections(pyop)
            self.tracker.update(self.detections)

    @staticmethod
    def _create_detections(pyop: PyOpenPoseNative, image_size: tuple):
        heatmaps = pyop.pose_heatmaps.copy()
        keypoints = pyop.pose_keypoints.copy()
        boxes = pyop.pose_bounding_box.copy()
        scores = pyop.pose_scores.copy()
        s_h = heatmaps.shape[0]/image_size[1]
        s_w = heatmaps.shape[1]/image_size[0]
        return create_detections(
            keypoints, scores, boxes, heatmaps, [s_w, s_h])

    @staticmethod
    def _create_detections(pyop: PyOpenPoseNative):
        boxes = pyop.pose_bounding_box.copy()
        scores = np.expand_dims(pyop.pose_scores.copy(), 1)
        return np.concatenate([boxes, scores], axis=1)

    def no_measurement_predict_and_update(self):
        if self.name == 'deep_sort':
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
