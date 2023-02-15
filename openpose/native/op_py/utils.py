import csv
import cv2
import json
import numpy as np
import os

from typing import Optional
from PIL import Image


# Based on:
# https://github.com/cheneeheng/realsense-simple-wrapper/blob/main/rs_py/rs_view_raw_data.py
def get_rs_sensor_dir(base_path: str, sensor: str) -> dict:
    path = {}
    for device in sorted(os.listdir(base_path)):
        path[device] = {}
        device_path = os.path.join(base_path, device)
        for ts in sorted(os.listdir(device_path)):
            path[device][ts] = os.path.join(base_path, device, ts, sensor)
    return path


# From:
# https://github.com/cheneeheng/realsense-simple-wrapper/blob/main/rs_py/rs_view_raw_data.py
def _get_brg_from_yuv(data_array: np.ndarray) -> np.ndarray:
    # Input: Intel handle to 16-bit YU/YV data
    # Output: BGR8
    UV = np.uint8(data_array >> 8)
    Y = np.uint8((data_array << 8) >> 8)
    YUV = np.zeros((data_array.shape[0], 2), 'uint8')
    YUV[:, 0] = Y
    YUV[:, 1] = UV
    YUV = YUV.reshape(-1, 1, 2)
    BGR = cv2.cvtColor(YUV, cv2.COLOR_YUV2BGR_YUYV)
    BGR = BGR.reshape(-1, 3)
    return BGR


# From:
# https://github.com/cheneeheng/realsense-simple-wrapper/blob/main/rs_py/rs_view_raw_data.py
def read_color_file(filename: str,
                    fileformat: Optional[str] = None) -> np.ndarray:
    if filename.endswith(('.png', '.jpeg', '.jpg')):
        image = np.array(Image.open(filename))[:, :, ::-1].copy()
    elif filename.endswith('.bin'):
        with open(filename, 'rb') as f:
            if fileformat.lower() in ['bgr8', 'rgb8']:
                image = np.fromfile(f, np.uint8)
            elif fileformat.lower() in ['yuyv']:
                image = np.fromfile(f, np.uint16)
                image = _get_brg_from_yuv(image.reshape(-1))
            else:
                raise ValueError("Unknown data type :", image.dtype)
    else:
        image = np.load(filename)
        if image.dtype == np.dtype(np.uint8):
            pass
        elif image.dtype == np.dtype(np.uint16):
            image = _get_brg_from_yuv(image.reshape(-1))
        else:
            raise ValueError("Unknown data type :", image.dtype)
    return image


# From:
# https://github.com/cheneeheng/realsense-simple-wrapper/blob/main/rs_py/rs_view_raw_data.py
def read_depth_file(filename: str) -> np.ndarray:
    if filename.endswith('.bin'):
        with open(filename, 'rb') as f:
            depth = np.fromfile(f, np.uint16)
    else:
        depth = np.fromfile(filename, np.uint16)
    return depth


# From:
# https://github.com/cheneeheng/realsense-simple-wrapper/blob/main/rs_py/rs_view_raw_data.py
def read_calib_file(calib_file: str) -> dict:
    calib_data = {}
    if calib_file.endswith('.json'):
        with open(calib_file, 'r') as f:
            calib_data = json.load(f)
    elif calib_file.endswith('.csv'):
        with open(calib_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    calib_data['color'] = {
                        'width': int(row[0]),
                        'height': int(row[1]),
                        'ppx': float(row[2]),
                        'ppy': float(row[3]),
                        'fx': float(row[4]),
                        'fy': float(row[5]),
                        'model': row[6],
                        'coeffs': [float(i) for i in row[7:12]],
                        'format': row[12],
                        'fps': int(row[13]),
                    }
                if line_count == 1:
                    calib_data['depth'] = {
                        'width': int(row[0]),
                        'height': int(row[1]),
                        'ppx': float(row[2]),
                        'ppy': float(row[3]),
                        'fx': float(row[4]),
                        'fy': float(row[5]),
                        'model': row[6],
                        'coeffs': [float(i) for i in row[7:12]],
                        'format': row[12],
                        'fps': int(row[13]),
                    }
                if line_count == 2:
                    calib_data['T_color_depth'] = {
                        'rotation': [float(i) for i in row[0:9]],
                        'translation': [float(i) for i in row[9:12]],
                    }
                if line_count == 3:
                    calib_data['depth']['depth_scale'] = float(row[0])
                    calib_data['depth']['depth_baseline'] = float(row[1])
                line_count += 1
    return calib_data


# https://github.com/nwojke/deep_sort/blob/280b8bdb255f223813ff4a8679f3e1321b08cdfc/deep_sort/nn_matching.py#L99
def cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


# https://www.geeksforgeeks.org/image-processing-without-opencv-python/
def resize_tensor(image, x, y):
    h, w, c = image.shape
    xScale = x/(w-1)
    yScale = y/(h-1)
    newImage = np.zeros([y, x, c])
    for i in range(y-1):
        for j in range(x-1):
            newImage[i + 1, j + 1] = image[1 + int(i / yScale),
                                           1 + int(j / xScale)]
    return newImage
