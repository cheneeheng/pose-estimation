import numpy as np

from .skeleton import PyOpenPoseNative
from .utils import Timer
from .utils_track import create_detections

from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracker.byte_tracker import STrack
from ByteTrack.yolox.tracker.byte_tracker import joint_stracks


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


# Tracking inspired by : https://github.com/ortegatron/liveposetracker
class Tracker():

    def __init__(self, mode: str = 'deep_sort', max_age: int = 30) -> None:
        """Intializes the tracker class.

        Args:
            max_age (int, optional): How long an untracked obj stays alive.
                Same as buffer_size in byte tracker. Defaults to 30.
        """
        if mode == 'byte_tracker':
            args = ByteTrackerArgs()
            args.track_buffer = max_age
            self.tracker = BYTETracker(args)
            self.name = 'byte_tracker'
        elif mode == 'deep_sort':
            args = DeepSortTrackerArgs()
            metric = NearestNeighborDistanceMetric(
                args.metric, args.matching_threshold, args. nn_budget)
            self.tracker = DeepSortTracker(metric, max_age=max_age)
            self.name = 'deep_sort'
        else:
            raise ValueError("Not implemented...")
        self.detections = None

    @property
    def tracks(self):
        if self.name == 'deep_sort':
            return self.tracker.tracks
        elif self.name == 'byte_tracker':
            return self.tracker.tracked_stracks

    def predict(self):
        if self.name == 'byte_tracker':
            raise ValueError("Not used...")
        self.tracker.predict()

    def update(self, pyop: PyOpenPoseNative, image_size: tuple = None):
        if self.name == 'deep_sort':
            self.detections = self._create_detections(pyop, image_size)
            self.tracker.update(self.detections)
        elif self.name == 'byte_tracker':
            boxes = pyop.pose_bounding_box.copy()
            scores = pyop.pose_scores.copy()
            detections = np.concatenate(
                [boxes, scores.reshape([len(scores), 1])], axis=1)
            self.tracker.update(detections)

    def _create_detections(self, pyop: PyOpenPoseNative, image_size: tuple):
        heatmaps = pyop.pose_heatmaps.copy()
        keypoints = pyop.pose_keypoints.copy()
        boxes = pyop.pose_bounding_box.copy()
        scores = pyop.pose_scores.copy()
        s_h = heatmaps.shape[0]/image_size[1]
        s_w = heatmaps.shape[1]/image_size[0]
        return create_detections(
            keypoints, scores, boxes, heatmaps, [s_w, s_h])

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
