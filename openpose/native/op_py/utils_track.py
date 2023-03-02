import numpy as np

from typing import Optional, Tuple

from .utils import resize_tensor
from submodules.StrongSORT.deep_sort.detection import Detection


# Tracking inspired by : https://github.com/ortegatron/liveposetracker
class PoseDetection(Detection):
    """
    Based on deep_sort/deep_sort/detection.py

    This class represents a bounding box detection in a single image.

    """

    def __init__(self,
                 pose: np.ndarray,
                 tlwh: np.ndarray,
                 confidence: float,
                 feature: np.ndarray,
                 ) -> None:
        self.pose = pose  # keypoints
        self.tlwh = tlwh  # bb from keypoints
        self.confidence = confidence  # pose score
        self.feature = feature  # heatmap from openpose
        self.feature_size = feature.shape  # heatmap from openpose


# baseb on https://github.com/nwojke/deep_sort/blob/master/deep_sort_app.py
def create_detections(keypoints: np.ndarray,
                      scores: np.ndarray,
                      bounding_boxes: np.ndarray,
                      heatmaps: np.ndarray,
                      heatmaps_scale: Tuple[float, float]) -> list:
    """Create detections for given frame index from the raw detection matrix.

    No NMS suppression of the boxes cause they are created from keypoints.

    """
    assert keypoints.shape[0] == scores.shape[0]
    assert keypoints.shape[0] == bounding_boxes.shape[0]

    s_w, s_h = heatmaps_scale

    detections = []
    for kpt, score, box in zip(keypoints, scores, bounding_boxes):
        # l, t, r, b = box
        ll = int(np.floor(box[0]*s_w))
        tt = int(np.floor(box[1]*s_h))
        rr = int(np.ceil(box[2]*s_w))
        bb = int(np.ceil(box[3]*s_h))
        feature = heatmaps[tt:bb, ll:rr]
        feature = resize_tensor(feature, 25, 50).flatten()
        tlhw = np.asarray([box[0], box[1], box[2] - box[0], box[3] - box[1]])
        detections.append(PoseDetection(kpt, tlhw, score, feature))

    return detections
