import cv2
import os
import time
from tqdm import trange

from openpose.native.python.skeleton import PyOpenPoseNative


def test_op_runtime():
    image_path = "openpose/data/test/pexels-photo-4384679.jpeg"
    target_path = "openpose/data/test/output/inference_native"
    model_folder = "/usr/local/src/openpose/models/"
    model_pose = "BODY_25"
    net_resolution = "-1x368"

    skel_thres = 0.5
    max_true_body = 2
    patch_offset = 2
    ntu_format = False

    os.makedirs(target_path, exist_ok=True)
    params = dict(
        model_folder=model_folder,
        model_pose=model_pose,
        net_resolution=net_resolution,
    )

    pyop = PyOpenPoseNative(params,
                            skel_thres,
                            max_true_body,
                            patch_offset,
                            ntu_format)
    pyop.initialize()

    t_total = 0
    N = 1000
    image = cv2.imread(image_path)
    image = cv2.resize(image, (384, 384))

    for _ in trange(N):
        t_start = time.time()
        pyop.predict(image)
        pyop.filter_prediction()
        # pyop.display(1, 'dummy')
        pyop.save_pose_keypoints(f'{target_path}/predictions.txt')
        t_total += time.time() - t_start

    print(f"Average inference time over {N} trials : {t_total/N}s")


if __name__ == "__main__":
    test_op_runtime()
    print(f"[INFO] : FINISHED")
