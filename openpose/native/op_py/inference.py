import argparse
import cv2
import os
import numpy as np

from typing import Optional


from openpose.native import PyOpenPoseNative
from openpose.native.op_py.args import get_parser
from openpose.native.op_py.utils_rs import get_rs_sensor_dir
from openpose.native.op_py.utils_rs import read_calib_file
from openpose.native.op_py.utils_rs import read_color_file_with_exception
from openpose.native.op_py.utils_rs import read_depth_file_with_exception
from openpose.native.op_py.utils_rs import prepare_save_paths


class Inferencer(object):

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

    def display(self, dev="1", image=None, bounding_box=False, tracks=None):
        if self.pyop.datum.poseScores is None:
            cv2.imshow(str(dev), image)
            cv2.waitKey(1000)
        else:
            self.pyop.display(str(dev),
                              scale=0.5,
                              bounding_box=bounding_box,
                              tracks=tracks)


def rs_offline_inference(args: argparse.Namespace):
    """Runs openpose offline by looking at images found in the image_path arg.

    Args:
        args (argparse.Namespace): CLI arguments
    """

    assert os.path.isdir(args.op_rs_dir), f'{args.op_rs_dir} does not exist...'

    OPI = Inferencer(args)

    base_path = args.op_rs_dir
    dev_trial_color_dir = get_rs_sensor_dir(base_path, 'color')
    dev_list = list(dev_trial_color_dir.keys())

    error_counter = 0
    error_state = False
    empty_counter = 0
    empty_state = False

    # 1. If no error
    while not error_state and not empty_state:

        # 2. loop through devices
        for dev, trial_color_dir in dev_trial_color_dir.items():

            # 3. loop through trials
            for trial, color_dir in trial_color_dir.items():

                color_files = sorted(os.listdir(color_dir))

                if len(color_files) == 0:
                    print(f"[INFO] : {color_dir} is empty...")
                    continue

                # 4. get the file that has not been inferred on.
                for i, color_file in enumerate(color_files):
                    color_filepath = os.path.join(color_dir, color_file)
                    skel_file = color_file.replace(color_file.split('.')[-1], 'csv')  # noqa
                    skel_dir = color_dir.replace('color', 'skeleton')
                    skel_filepath = os.path.join(skel_dir, skel_file)
                    if not os.path.exists(skel_filepath):
                        i = i - 1
                        break

                if i + 1 == len(color_files):
                    print(f"[INFO] : {color_dir} is fully evaluated...")
                    empty_counter += 1
                    if empty_counter > 300:
                        print("[INFO] Retried 300 times and no new files...")
                        empty_state = True
                    continue

                print(f"[INFO] : {len(color_files)-  max(i, 0)} files left...")

                # 5. get the color image
                image, error_state, error_counter = \
                    read_color_file_with_exception(
                        error_state, error_counter, color_filepath)
                if image is None:
                    continue

                # 5b. get the depth image
                depth = None
                if args.op_rs_extract_3d_skel:
                    depth_filepath = color_filepath.replace('color', 'depth')
                    depth, error_state, error_counter = \
                        read_depth_file_with_exception(
                            error_state, error_counter, depth_filepath)
                    if depth is None:
                        continue

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
                        if args.op_rs_extract_3d_skel:
                            main_dir = os.path.dirname(os.path.dirname(kpt_save_path))  # noqa
                            intr_mat = read_calib_file(main_dir + "/calib/calib.csv")  # noqa
                            OPI.predict_3d(image,
                                           depth,
                                           intr_mat,
                                           kpt_save_path,
                                           kpt_3d_save_path,
                                           skel_image_save_path)
                        else:
                            OPI.predict(image,
                                        kpt_save_path,
                                        skel_image_save_path)

                except Exception as e:
                    print(e)
                    continue

                if args.op_rs_delete_image:
                    os.remove(color_filepath)


if __name__ == "__main__":

    [arg_op, _] = get_parser().parse_known_args()
    rs_offline_inference(arg_op)

    print(f"[INFO] : FINISHED")
