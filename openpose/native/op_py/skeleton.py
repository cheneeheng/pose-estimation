import argparse
import cv2
import os
import numpy as np
import pyopenpose as op

from typing import Optional, Union, List, Tuple


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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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


class PyOpenPoseNative:
    """Wrapper for the native openpose python interface. """

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
            # params["heatmaps_add_parts"] = True
            # params["heatmaps_add_bkg"] = True
            # params["heatmaps_add_PAFs"] = True
            # params["heatmaps_scale"] = 2

        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.datum = op.Datum()

        # results
        self._pose_scores = None
        self._pose_heatmaps = None
        self._pose_keypoints = None
        self._pose_keypoints_3d = None
        self._pose_bounding_box = None
        self._pose_bounding_box_int = None
        self.reset()

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
        if self._pose_scores is None:
            return self.datum.poseScores
        else:
            return self._pose_scores

    @property
    def pose_heatmaps(self) -> Optional[np.ndarray]:
        # [op_h, op_w, 76] : all heatmaps
        # [M, J, 3] : person, joints, xyz
        if self._pose_heatmaps is None:
            return np.moveaxis(self.datum.poseHeatMaps, 0, -1)
        else:
            return self._pose_heatmaps

    @property
    def pose_keypoints(self) -> Optional[np.ndarray]:
        # [M, V, C]; M = Subjects; V = Joints; C = (x,y,score)
        if self._pose_keypoints is None:
            return self.datum.poseKeypoints
        else:
            return self._pose_keypoints

    @property
    def pose_keypoints_3d(self) -> Optional[np.ndarray]:
        # [M, V, C]; C = (x,y,z)
        return self._pose_keypoints_3d

    @property
    def pose_bounding_box(self) -> np.ndarray:
        if self._pose_bounding_box is None:
            bb = []
            for pose_keypoints in self.pose_keypoints:
                u = pose_keypoints[:, 0]
                v = pose_keypoints[:, 1]
                s = pose_keypoints[:, 2]
                u_min = u[s != 0].min()
                u_max = u[s != 0].max()
                v_min = v[s != 0].min()
                v_max = v[s != 0].max()
                bb.append(np.asarray([u_min, v_min, u_max, v_max]))
            self._pose_bounding_box = np.stack(bb)
        return self._pose_bounding_box

    @property
    def pose_bounding_box_int(self) -> np.ndarray:
        if self._pose_bounding_box_int is None:
            _bb = self.pose_bounding_box
            u_min = np.floor(_bb[:, 0]).astype(int)
            v_min = np.floor(_bb[:, 1]).astype(int)
            u_max = np.ceil(_bb[:, 2]).astype(int)
            v_max = np.ceil(_bb[:, 3]).astype(int)
            self._pose_bounding_box_int = np.stack(
                [u_min, v_min, u_max, v_max], axis=1)
        return self._pose_bounding_box_int

    def reset(self) -> None:
        self._pose_scores = None
        self._pose_heatmaps = None
        self._pose_keypoints = None
        self._pose_keypoints_3d = None
        self._pose_bounding_box = None
        self._pose_bounding_box_int = None
        return

    def configure(self, params: dict = None) -> None:
        if params is not None:
            self.opWrapper.configure(params)
        return

    def initialize(self) -> None:
        # Starting OpenPose
        self.opWrapper.start()
        return

    def predict(self, image: np.ndarray) -> None:
        self.reset()
        self.datum.cvInputData = image
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        return

    def filter_prediction(self) -> None:
        scores = self.pose_scores
        # 1. Empty array if scores is None (no skeleton at all)
        if scores is None:
            print("No skeleton detected...")
            self.reset()
        # 2. Else pick pose based on prediction scores
        else:
            scores_filtered = []
            keypoints_filtered = []
            max_score_idxs = np.argsort(scores)[-self.max_true_body:]
            for max_score_idx in max_score_idxs:
                if scores[max_score_idx] < self.skel_thres:
                    # print(f"Low skeleton score {scores[max_score_idx]:.2f}, "
                    #       f"skip skeleton...")
                    continue
                else:
                    keypoint = self.pose_keypoints[max_score_idx]
                    keypoints_filtered.append(keypoint)
                    scores_filtered.append(scores[max_score_idx])
            if len(scores_filtered) == 0:
                self.reset()
            else:
                # [M, V, C]; M = Subjects; V = Joints; C = (x,y,score)
                self._pose_keypoints = np.stack(keypoints_filtered, axis=0)
                # [M]
                self._pose_scores = np.stack(scores_filtered, axis=0)

            print(f"Skeletons filtered: {len(keypoints_filtered)}/{len(max_score_idxs)}")  # noqa

        return

    def convert_to_3d(self,
                      depth_image: np.ndarray,
                      intr_mat: Union[list, np.ndarray],
                      depth_scale: float = 1e-3,
                      ) -> None:
        if self.pose_keypoints is None:
            print("No skeleton detected...")
            pass
        else:
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
            self._pose_keypoints_3d = np.asarray(pose_keypoints_3d)
        return

    def save_pose_keypoints(self, save_path: str) -> None:
        save_2d_skeleton(keypoints=self.pose_keypoints,
                         scores=self.pose_scores,
                         save_path=save_path)
        return

    def save_3d_pose_keypoints(self, save_path: str) -> None:
        save_3d_skeleton(keypoints=self.pose_keypoints_3d,
                         scores=self.pose_scores,
                         save_path=save_path)
        return

    def _draw_skeleton_image(self,
                             depth_image: Optional[np.ndarray] = None,
                             ) -> np.ndarray:
        keypoint_image = self.opencv_image
        # keypoint_image = cv2.flip(keypoint_image, 1)
        cv2.putText(keypoint_image,
                    "KP (%) : " + str(round(np.mean(self.pose_scores), 2)),
                    (10, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA)
        if depth_image is not None:
            colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.065, beta=0),
                cv2.COLORMAP_INFERNO
            )
            # colormap = cv2.flip(colormap, 1)
            keypoint_image = cv2.addWeighted(
                keypoint_image, 0.7, colormap, 0.7, 0)
            # overlay = cv2.resize(overlay, (800, 450))
        return keypoint_image

    def display(self,
                device_sn: str,
                speed: int = 1000,
                scale: float = 1.0,
                depth_image: Optional[np.ndarray] = None,
                bounding_box: bool = False,
                tracks: Optional[list] = None) -> bool:
        image = self._draw_skeleton_image(depth_image)

        if bounding_box:
            for idx, bb in enumerate(self.pose_bounding_box_int):
                tl, br = bb[0:2], bb[2:4]
                image = cv2.rectangle(image, tl, br, (0, 255, 0), 2)

        if tracks is not None:
            for track in tracks:
                try:
                    # deepsort / ocsort
                    bb = track.to_tlbr()
                except AttributeError:
                    # bytetrack
                    bb = track.tlbr
                l, t, r, b = bb
                tl = (np.floor(l).astype(int), np.floor(t).astype(int))
                br = (np.ceil(r).astype(int), np.ceil(b).astype(int))
                image = cv2.rectangle(image, tl, br, (0, 0, 255), 2)
                cv2.putText(image,
                            f"ID : {track.track_id}",
                            tl,
                            cv2.FONT_HERSHEY_PLAIN,
                            2,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA)

        image = cv2.resize(image, (int(image.shape[1]*scale),
                                   int(image.shape[0]*scale)))

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
        key = cv2.waitKey(speed)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            cv2.waitKey(5)
            return True
        else:
            return False

    def save_skeleton_image(self,
                            save_path: str,
                            scale: int = 1,
                            depth: Optional[np.ndarray] = None) -> None:
        image = self._draw_skeleton_image(scale, depth)
        # image = cv2.flip(image, 1)
        # _path = save_path.replace('skeleton', 'skeleton_color')
        # _path = _path.split('.')[0] + '.jpg'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image)
        return


class OpenPosePoseExtractor:
    """A high level wrapper for the `PyOpenPoseNative` class. This class only
    has functions that intialize openpose, infer + save pose, and display pose.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.pyop = PyOpenPoseNative(
            dict(
                model_folder=args.op_model_folder,
                model_pose=args.op_model_pose,
                net_resolution=args.op_net_resolution,
                heatmaps_add_parts=args.op_heatmaps_add_parts,
                heatmaps_add_bkg=args.op_heatmaps_add_bkg,
                heatmaps_add_PAFs=args.op_heatmaps_add_PAFs,
                heatmaps_scale=args.op_heatmaps_scale,
            ),
            args.op_skel_thres,
            args.op_max_true_body,
            args.op_patch_offset,
            args.op_ntu_format
        )
        self.pyop.initialize()

    def predict(self,
                image: np.ndarray,
                kpt_save_path: Optional[str] = None,
                skel_image_save_path: Optional[str] = None) -> None:
        self.pyop.predict(image)
        self.pyop.filter_prediction()
        if kpt_save_path is not None:
            self.pyop.save_pose_keypoints(kpt_save_path)
        if skel_image_save_path is not None:
            self.pyop.save_skeleton_image(skel_image_save_path)
        # print(f"[INFO] : Openpose output saved in {kpt_save_path}")

    def predict_3d(self,
                   image: np.ndarray,
                   depth: np.ndarray,
                   intr_mat: np.ndarray,
                   depth_scale: float = 1e-3,
                   kpt_save_path: Optional[str] = None,
                   kpt_3d_save_path: Optional[str] = None,
                   skel_image_save_path: Optional[str] = None) -> None:
        self.predict(image, kpt_save_path, skel_image_save_path)
        self.pyop.convert_to_3d(
            depth_image=depth,
            intr_mat=intr_mat,
            depth_scale=depth_scale
        )
        if kpt_3d_save_path is not None:
            self.pyop.save_3d_pose_keypoints(kpt_3d_save_path)

    def display(self,
                dev: str = "1",
                speed: int = 1000,
                scale: int = 1.0,
                image: Optional[np.ndarray] = None,
                bounding_box: bool = False,
                tracks=None):
        if self.pyop.datum.poseScores is None:
            cv2.imshow(str(dev), image)
            cv2.waitKey(speed)
        else:
            _image = cv2.resize(image, (int(image.shape[1]*scale),
                                        int(image.shape[0]*scale)))
            cv2.imshow(str(dev)+"_ori", _image)
            self.pyop.display(str(dev),
                              scale=scale,
                              speed=speed,
                              bounding_box=bounding_box,
                              tracks=tracks)
