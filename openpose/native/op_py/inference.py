import argparse
import cv2
import os
import numpy as np

from typing import Optional

from openpose.native import PyOpenPoseNative
from openpose.native.op_py.args import get_parser
from openpose.native.op_py.utils import Timer
from openpose.native.op_py.utils_rs import get_rs_sensor_dir
from openpose.native.op_py.utils_rs import read_calib_file
from openpose.native.op_py.utils_rs import read_color_file_with_exception
from openpose.native.op_py.utils_rs import read_depth_file_with_exception
from openpose.native.op_py.utils_rs import prepare_save_paths
from openpose.native.op_py.utils_track import create_detections

from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric


class PoseExtractor(object):

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
                image: Optional[np.ndarray] = None,
                bounding_box: bool = False,
                tracks=None):
        if self.pyop.datum.poseScores is None:
            cv2.imshow(str(dev), image)
            cv2.waitKey(speed)
        else:
            cv2.imshow(str(dev)+"_ori", image)
            self.pyop.display(str(dev),
                              scale=1.0,
                              speed=speed,
                              bounding_box=bounding_box,
                              tracks=tracks)


# Tracking inspired by : https://github.com/ortegatron/liveposetracker
class Tracker():

    def __init__(self, max_age=30) -> None:
        max_cosine_distance = 0.2
        nn_budget = None
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric, max_age=max_age)
        self.detections = None

    def predict(self):
        self.tracker.predict()

    def update(self, pyop: PyOpenPoseNative, image_size: tuple):
        self.detections = self._create_detections(pyop, image_size)
        self.tracker.update(self.detections)

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


def rs_offline_inference(args: argparse.Namespace):
    """Runs openpose offline by looking at images found in the image_path arg.

    Args:
        args (argparse.Namespace): CLI arguments
    """

    assert os.path.isdir(args.op_rs_dir), f'{args.op_rs_dir} does not exist...'

    base_path = args.op_rs_dir
    dev_trial_color_dir = get_rs_sensor_dir(base_path, 'color')
    dev_list = list(dev_trial_color_dir.keys())

    error_counter = 0
    error_state = False
    empty_counter = 0
    empty_state = False
    delay_switch = 3
    delay_counter = 0

    c = 30

    PE = PoseExtractor(args)
    TK = Tracker(30//delay_switch) if args.op_track else None

    # 1. If no error
    while not error_state and not empty_state:

        # 2. loop through devices
        for dev, trial_color_dir in dev_trial_color_dir.items():

            # 3. loop through trials
            for trial, color_dir in trial_color_dir.items():

                # color_files = sorted(os.listdir(color_dir))

                # if len(color_files) == 0:
                #     print(f"[INFO] : {color_dir} is empty...")
                #     continue

                # # 4. get the file that has not been inferred on.
                # for i, color_file in enumerate(color_files):
                #     color_filepath = os.path.join(color_dir, color_file)
                #     skel_file = color_file.replace(color_file.split('.')[-1], 'csv')  # noqa
                #     skel_dir = color_dir.replace('color', 'skeleton')
                #     skel_filepath = os.path.join(skel_dir, skel_file)
                #     if not os.path.exists(skel_filepath):
                #         i = i - 1
                #         break

                # if i + 1 == len(color_files):
                #     print(f"[INFO] : {color_dir} is fully evaluated...")
                #     empty_counter += 1
                #     if empty_counter > 300:
                #         print("[INFO] Retried 300 times and no new files...")
                #         empty_state = True
                #     continue

                # print(
                #     f"[INFO] : {len(color_files)- max(i+1, 0)} files left...")

                with Timer(f"{c} - file path selection"):
                    color_files = sorted(os.listdir(color_dir))
                    color_filepath = os.path.join(color_dir, color_files[c])
                    skel_file = color_files[c].replace(color_files[c].split('.')[-1], 'csv')  # noqa
                    skel_dir = color_dir.replace('color', 'skeleton')
                    skel_filepath = os.path.join(skel_dir, skel_file)
                    c += 1

                    # 5. get the color image
                    image, error_state, error_counter = \
                        read_color_file_with_exception(
                            error_state, error_counter, color_filepath)
                    if image is None:
                        continue

                    # 5b. get the depth image
                    depth = None
                    if args.op_rs_extract_3d_skel:
                        depth_filepath = color_filepath.replace('color',
                                                                'depth')
                        depth, error_state, error_counter = \
                            read_depth_file_with_exception(
                                error_state, error_counter, depth_filepath)
                        if depth is None:
                            continue

                with Timer("data preparation"):
                    # 6. reshape images
                    try:
                        image = image.reshape(args.op_rs_image_height,
                                              args.op_rs_image_width, 3)
                        save_path_list = prepare_save_paths(
                            skel_filepath,
                            args.op_rs_save_skel,
                            args.op_rs_save_skel_image,
                            args.op_rs_save_3d_skel)
                        data_tuples = [[image] + save_path_list]

                    except Exception as e:
                        print(e)
                        print("Stacked data detected")
                        print(f"Current image shape : {image.shape}")

                        try:
                            image = image.reshape(args.op_rs_image_height,
                                                  args.op_rs_image_width*3, 3)
                        except Exception as e:
                            print(e)
                            continue

                        data_tuples = []
                        for i in range(3):
                            _dir = skel_dir.replace(dev_list[0], dev_list[i])
                            _path = skel_file.split('.')[0]
                            _path = _path.split('_')[i]
                            skel_filepath = os.path.join(_dir, _path + '.csv')
                            save_path_list = prepare_save_paths(
                                skel_filepath,
                                args.op_rs_save_skel,
                                args.op_rs_save_skel_image,
                                args.op_rs_save_3d_skel)
                            j = args.op_rs_image_width * i
                            k = args.op_rs_image_width * (i+1)
                            img = image[:, j:k, :]
                            data_tuples.append([img] + save_path_list)

                # 7. infer images
                try:
                    for (image,
                         kpt_save_path,
                         skel_image_save_path,
                         kpt_3d_save_path) in data_tuples:

                        if TK is not None:
                            if delay_counter > 1:
                                delay_counter -= 1
                                if kpt_save_path is not None:
                                    open(kpt_save_path, 'a').close()
                                TK.no_measurement_predict_and_update()
                                PE.display(dev, 1, image, True,
                                           TK.tracker.tracks)
                                continue
                            else:
                                delay_counter = delay_switch

                        with Timer("infer"):
                            if args.op_rs_extract_3d_skel:
                                main_dir = os.path.dirname(os.path.dirname(kpt_save_path))  # noqa
                                intr_mat = read_calib_file(main_dir + "/calib/calib.csv")  # noqa
                                PE.predict_3d(image,
                                              depth,
                                              intr_mat,
                                              kpt_save_path,
                                              kpt_3d_save_path,
                                              skel_image_save_path)
                            else:
                                PE.predict(image,
                                           kpt_save_path,
                                           skel_image_save_path)

                        if TK is not None:
                            with Timer("predict"):
                                TK.predict()
                            with Timer("update"):
                                TK.update(PE.pyop, (args.op_rs_image_width,
                                                    args.op_rs_image_height))

                        PE.display(dev, 3000, image, True, TK.tracker.tracks)
                        print(f"Number of tracks : {len(TK.tracker.tracks)}")

                except Exception as e:
                    print(e)
                    continue

                if args.op_rs_delete_image:
                    os.remove(color_filepath)


if __name__ == "__main__":

    [arg_op, _] = get_parser().parse_known_args()
    rs_offline_inference(arg_op)

    print(f"[INFO] : FINISHED")
