import argparse
import cv2
import numpy as np
import os

from openpose.native import OpenposeStoragePaths
from openpose.native import PyOpenPoseNative
from openpose.native import get_op_parser
from realsense import get_rs_parser
from realsense import initialize_rs_devices

from infer.inference import parse_arg
from infer.inference import get_parser as get_agcn_parser
from infer.inference import init_file_and_folders
from infer.inference import init_preprocessor
from infer.inference import init_model
from infer.inference import model_inference
from infer.inference import filter_logits


class MovingAverage:
    def __init__(self, length: int) -> None:
        self.len = length
        self.mva_list = []

    def fit(self, data: np.ndarray):
        if len(self.mva_list) < self.len:
            self.mva_list.append(data)
            logits = data
        else:
            self.mva_list = self.mva_list[1:] + [data]
            logits = np.mean(np.stack(self.mva_list, axis=0), axis=0)
            self.mva_list[-1] = logits
        return logits


if __name__ == "__main__":

    [arg_rs, _] = get_rs_parser().parse_known_args()
    [arg_op, _] = get_op_parser().parse_known_args()
    [arg_agcn, _] = parse_arg(get_agcn_parser())

    # 0. Initialize ------------------------------------------------------------
    # AAGCN
    MAPPING, _, output_dir = init_file_and_folders(arg_agcn)
    DataProc = init_preprocessor(arg_agcn)
    Model = init_model(arg_agcn)

    # OPENPOSE
    params = dict(
        model_folder=arg_op.op_model_folder,
        model_pose=arg_op.op_model_pose,
        net_resolution=arg_op.net_resolution,
        disable_blending=True
    )
    pyop = PyOpenPoseNative(params,
                            arg_op.op_skel_thres,
                            arg_op.op_max_true_body,
                            arg_op.op_patch_offset,
                            arg_op.op_ntu_format)
    pyop.initialize()

    # REALSENSE
    rsw = initialize_rs_devices(arg_rs, OpenposeStoragePaths)

    state = True
    empty_skeleton_3d = np.zeros((25, 3))

    mva = MovingAverage(5)

    print("Starting frame capture loop...")
    try:
        while state:

            # 1. Get rs data ---------------------------------------------------
            print("Running...")
            frames = rsw.run(display=arg_rs.display_frame)
            if not len(frames) > 0:
                continue

            for dev_sn, data_dict in frames.items():
                color_image = frames[dev_sn]['color']
                depth_image = frames[dev_sn]['depth']
                timestamp = frames[dev_sn]['timestamp']
                calib = frames[dev_sn]['calib']

                # 2. Predict pose ----------------------------------------------
                # bgr format
                pyop.predict(color_image)

                # 3. Save data -------------------------------------------------
                if arg_op.op_save_skel:
                    intr_mat = calib['color'][0]['intrinsic_mat']
                    skel_save_path = os.path.join(
                        rsw.storage_paths[dev_sn].skeleton,
                        f'{timestamp:020d}' + '.txt'
                    )
                    pyop.convert_to_3d(
                        intr_mat=intr_mat,
                        depth_image=depth_image,
                        empty_pose_keypoints_3d=empty_skeleton_3d,
                        save_path=skel_save_path
                    )

                if arg_op.op_display_depth > 0:
                    stop = pyop.display(arg_op.op_display_depth,
                                        dev_sn, depth_image)
                    if stop:
                        break
                elif arg_op.op_display > 0:
                    stop = pyop.display(arg_op.op_display,
                                        dev_sn, depth_image)
                    if stop:
                        break

                # 4. Action recognition ----------------------------------------
                # 4.1. Batch frames to fixed length.
                skel_data = np.stack(skel_data, axis=0)  # m,v,c
                skel_data = np.expand_dims(
                    skel_data[:, :arg_agcn.num_joint, :],
                    axis=1
                )  # m,t,v,c
                DataProc.append_data(skel_data)
                try:
                    input_data = DataProc.select_skeletons_and_normalize_data(
                        arg_agcn.max_num_skeleton_true,
                        sgn='sgn' in arg_agcn.model
                    )
                    input_data = DataProc.data

                    if np.sum(input_data) == np.nan:
                        DataProc.clear_data_array()
                        input_data = DataProc.data

                    # 4.2. repeat segments in a sequence.
                    if 'aagcn' in arg_agcn.model:
                        # N, C, T, V, M
                        input_data = np.concatenate(
                            [input_data, input_data, input_data], axis=2)

                    # 4.3. Inference.
                    logits, preds = model_inference(arg_agcn, Model, input_data)  # noqa
                    logits = mva.fit(logits[0].numpy())
                    logits, preds = logits.tolist(), preds.item()
                    sort_idx, new_logits = filter_logits(logits)

                    output_file = os.path.join(
                        output_dir, f'{timestamp:020d}' + '.txt')
                    with open(output_file, 'a+') as f:
                        output_str1 = ",".join([str(i) for i in sort_idx])
                        output_str2 = ",".join([str(i) for i in new_logits])
                        output_str = f'{output_str1};{output_str2}\n'
                        output_str = output_str.replace(
                            '[', '').replace(']', '')
                        f.write(output_str)
                        if len(sort_idx) > 0:
                            print(f"Original Pred: {preds}, Filtered Pred: {sort_idx[0]: >2}, Logit: {new_logits[0]*100:>5.2f}")  # noqa
                        else:
                            print(preds)

                except Exception as e:
                    print(e)
                    print("Inference error...")

    except Exception as e:
        print(e)
        print("Stopping realsense...")
        rsw.stop()

    finally:
        rsw.stop()
