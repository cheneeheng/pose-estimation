import numpy as np
import pyopenpose as op


class PyOpenPoseNative(object):

    def __init__(self, params: dict = None) -> None:
        super().__init__()

        # default parameters
        if params is None:
            params = dict()
            params["model_folder"] = "/usr/local/src/openpose/models/"
            params["model_pose"] = "BODY_25"
            params["net_resolution"] = "-1x368"

        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.datum = op.Datum()

    def configure(self, params: dict = None):
        if params is not None:
            self.opWrapper.configure(params)

    def initialize(self):
        # Starting OpenPose
        self.opWrapper.start()

    def predict(self, image):
        self.datum.cvInputData = image
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))

    @property
    def opencv_image(self):
        return self.datum.cvOutputData

    @property
    def pose_scores(self) -> list:
        return self.datum.poseScores

    @property
    def pose_keypoints(self) -> list:
        return self.datum.poseKeypoints


def get_3d_skeleton(skeleton, depth_img, intr_mat, patch_offset=2):
    H, W = depth_img.shape
    fx = intr_mat[0, 0]
    fy = intr_mat[1, 1]
    cx = intr_mat[0, 2]
    cy = intr_mat[1, 2]
    joints3d = []
    for x, y, _ in skeleton:
        patch = depth_img[
            max(0, int(y-patch_offset)):min(H, int(y+patch_offset)),  # noqa
            max(0, int(x-patch_offset)):min(W, int(x+patch_offset))  # noqa
        ]
        depth_avg = np.mean(patch)
        x3d = (x-cx) / fx * depth_avg
        y3d = (y-cy) / fy * depth_avg
        joints3d.append([x3d, y3d, depth_avg])
    return np.array(joints3d)
