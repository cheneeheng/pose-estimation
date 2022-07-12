import cv2
import numpy as np
import pyopenpose as op

from typing import Optional, Union, List


class PyOpenPoseNative:

    def __init__(self,
                 params: Optional[dict] = None,
                 skel_thres: float = 0.0,
                 max_true_body: int = 2,
                 patch_offset: int = 2,
                 ntu_format: bool = False) -> None:
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
        self.__pose_scores = None
        self.__pose_keypoints = None
        self.__pose_keypoints_3d = None

        # for 3d skel
        self.skel_thres = skel_thres
        self.max_true_body = max_true_body
        self.patch_offset = patch_offset
        self.ntu_format = ntu_format

    @property
    def opencv_image(self) -> np.ndarray:
        # [H, W, C]
        return self.datum.cvOutputData

    @property
    def pose_scores(self) -> Optional[np.ndarray]:
        # [M]
        if self.__pose_scores is None:
            return self.datum.poseScores
        else:
            return self.__pose_scores

    @property
    def pose_keypoints(self) -> Optional[np.ndarray]:
        # [M, V, C]; C = (x,y,score)
        if self.__pose_keypoints is None:
            return self.datum.poseKeypoints
        else:
            return self.__pose_keypoints

    def configure(self, params: dict = None) -> None:
        if params is not None:
            self.opWrapper.configure(params)

    def initialize(self) -> None:
        # Starting OpenPose
        self.opWrapper.start()

    def predict(self, image: np.ndarray) -> None:
        self.datum.cvInputData = image
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        self.__pose_keypoints = None
        self.__pose_scores = None

    def filter_prediction(self) -> None:
        scores = self.pose_scores
        # 3.a. Empty array if scores is None (no skeleton at all)
        # 3.b. Else pick pose based on prediction scores
        if scores is None:
            print("No skeleton detected...")
            self.__pose_keypoints = None
            self.__pose_scores = None
        else:
            scores_filtered = []
            keypoints_filtered = []
            max_score_idxs = np.argsort(scores)[-self.max_true_body:]
            for max_score_idx in max_score_idxs:
                if scores[max_score_idx] < self.skel_thres:
                    print("Low skeleton score, skip skeleton...")
                else:
                    keypoint = self.pose_keypoints[max_score_idx]
                    keypoints_filtered.append(keypoint)
                    scores_filtered.append(max_score_idx)
            if len(scores_filtered) == 0:
                self.__pose_keypoints = None
                self.__pose_scores = None
            else:
                # [M, V, C]; C = (x,y,score)
                self.__pose_keypoints = np.stack(keypoints_filtered, axis=0)
                # [M]
                self.__pose_scores = np.stack(scores_filtered, axis=0)

    def convert_to_3d(self,
                      depth_image: np.ndarray,
                      intr_mat: Union[list, np.ndarray],
                      depth_scale: float = 1e-3,
                      ) -> List[np.ndarray]:
        pose_keypoints_3d = []
        # 3.a. Empty array if scores is None (no skeleton at all)
        # 3.b. Else pick pose based on prediction scores
        for pose_keypoint in self.pose_keypoints:
            # ntu_format => x,y(up),z(neg) in meter.
            # [V,C]
            skeleton_3d = get_3d_skeleton(
                skeleton=pose_keypoint,
                depth_img=depth_image,
                intr_mat=intr_mat,
                depth_scale=depth_scale,
                patch_offset=self.patch_offset,
                ntu_format=self.ntu_format
            )
            pose_keypoints_3d.append(skeleton_3d)
        self.__pose_keypoints_3d = np.array(pose_keypoints_3d)

    def save_pose_keypoints(self, save_path: str) -> None:
        save_2d_skeleton(keypoints=self.pose_keypoints,
                         scores=self.pose_scores,
                         save_path=save_path)

    def save_3d_pose_keypoints(self, save_path: str) -> None:
        save_3d_skeleton(keypoints=self.__pose_keypoints_3d,
                         scores=self.pose_scores,
                         save_path=save_path)

    def __draw_skeleton_image(self,
                              scale: int,
                              depth_image: Optional[np.ndarray] = None
                              ) -> np.ndarray:
        keypoint_image = self.opencv_image
        keypoint_image = cv2.flip(keypoint_image, 1)
        if scale < 1000:
            keypoint_image = cv2.resize(keypoint_image,
                                        (keypoint_image.shape[1]//scale,
                                         keypoint_image.shape[0]//scale))
        cv2.putText(keypoint_image,
                    "KP (%) : " + str(round(max(self.pose_scores), 2)),
                    (10, 20),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA)
        if depth_image is not None:
            colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.065, beta=0),
                cv2.COLORMAP_INFERNO
            )
            colormap = cv2.flip(colormap, 1)
            keypoint_image = cv2.addWeighted(
                keypoint_image, 0.7, colormap, 0.7, 0)
            # overlay = cv2.resize(overlay, (800, 450))
        return keypoint_image

    def display(self,
                scale: int,
                device_sn: str,
                depth_image: Optional[np.ndarray] = None) -> bool:
        image = self.__draw_skeleton_image(scale, depth_image)
        # overlay = cv2.resize(overlay, (800, 450))
        if depth_image is None:
            win_name = f'keypoint_image_{device_sn}'
        else:
            win_name = f'keypoint_depth_image_{device_sn}'
        if scale >= 1000:
            cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(win_name,
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        cv2.imshow(win_name, image)
        # cv2.moveWindow("depth_keypoint_overlay", 1500, 300)
        key = cv2.waitKey(30)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            cv2.waitKey(5)
            return True
        else:
            return False


def get_3d_skeleton(skeleton: np.ndarray,
                    depth_img: np.ndarray,
                    intr_mat: Union[list, np.ndarray],
                    depth_scale: float = 1e-3,
                    patch_offset: int = 2,
                    ntu_format: bool = False) -> np.ndarray:
    if isinstance(intr_mat, list):
        fx = intr_mat[0]
        fy = intr_mat[4]
        cx = intr_mat[2]
        cy = intr_mat[5]
    elif isinstance(intr_mat, np.ndarray):
        fx = intr_mat[0, 0]
        fy = intr_mat[1, 1]
        cx = intr_mat[0, 2]
        cy = intr_mat[1, 2]
    else:
        raise ValueError("Unknown intr_mat format.")
    H, W = depth_img.shape
    joints3d = []
    for x, y, _ in skeleton:
        patch = depth_img[
            max(0, int(y-patch_offset)):min(H, int(y+patch_offset)),  # noqa
            max(0, int(x-patch_offset)):min(W, int(x+patch_offset))  # noqa
        ]
        depth_avg = np.mean(patch)
        x3d = (x-cx) / fx * depth_avg
        y3d = (y-cy) / fy * depth_avg
        if ntu_format:
            joints3d.append([-x3d*depth_scale, -depth_avg*depth_scale,
                             -y3d*depth_scale])
        else:
            joints3d.append([x3d, y3d, depth_avg])
    return np.array(joints3d)


def save_2d_skeleton(keypoints: Optional[np.ndarray],
                     scores: Optional[np.ndarray],
                     save_path: str) -> None:
    # keypoints: [M, V, C]; C = (x,y,score)
    # scores: [M]
    if keypoints is None:
        open(save_path, 'a').close()
    else:
        M, _, _ = keypoints.shape
        data = np.concatenate([scores.reshape((M, 1)),
                               keypoints.reshape((M, -1))], axis=1)
        np.savetxt(save_path, data, delimiter=',')


def save_3d_skeleton(keypoints: Optional[np.ndarray],
                     scores: Optional[np.ndarray],
                     save_path: str) -> None:
    # keypoints: [M, V, C]; C = (x,y,z)
    # scores: [M]
    return save_2d_skeleton(keypoints, scores, save_path)
