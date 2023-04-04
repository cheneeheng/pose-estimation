import cv2
import os
import time
from tqdm import trange

from openpose.native.python.skeleton import PyOpenPoseNative


def test_op_runtime():
    image_path = "data/test/pexels-photo-4384679.jpeg"
    target_path = "data/test/output/inference_native"
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
    N = 500
    image = cv2.imread(image_path)
    image = cv2.resize(image, (384, 384))

    for n in trange(N):
        t_start = time.time()
        pyop.predict(image)
        pyop.filter_prediction()
        # print(pyop.pose_scores)
        # pyop.display(1, 'dummy')
        # pyop.save_pose_keypoints(f'{target_path}/predictions.txt')
        if n > N//2:
            t_total += time.time() - t_start

    print(f"\nAverage inference time over {N-(N//2)} trials : {t_total/(N-(N//2)):.6f}s or {(N-(N//2))/t_total:.6f}fps")  # noqa
    print(pyop.pose_scores)


if __name__ == "__main__":
    test_op_runtime()
    print(f"[INFO] : FINISHED")
