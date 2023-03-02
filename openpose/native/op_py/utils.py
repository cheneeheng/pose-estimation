import argparse
import time

import numpy as np


class Timer:
    def __init__(self, text: str = '', enable: bool = True) -> None:
        self._t_start = None
        self._text = text
        self._enable = enable

    def __enter__(self) -> None:
        if self._enable:
            self._t_start = time.time()

    def __exit__(self, *args) -> None:
        if self._enable:
            _duration = time.time() - self._t_start
            print(f"[INFO] : Timer >>> "
                  f"{self._text: <40} >>> "
                  f"{_duration:.6f} s")


class Error:
    def __init__(self):
        self.state = False
        self.counter = 0

    def reset(self):
        self.state = False
        self.counter = 0


def str2bool(v: str) -> bool:
    """Change string to boolean.

    Args:
        v (str): Boolean or 0/1 in string format.

    Raises:
        argparse.ArgumentTypeError: Boolean value or 0/1 expected.

    Returns:
        bool: Either true or false
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# https://github.com/nwojke/deep_sort/blob/280b8bdb255f223813ff4a8679f3e1321b08cdfc/deep_sort/nn_matching.py#L99
def cosine_distance(a: np.ndarray,
                    b: np.ndarray,
                    data_is_normalized: bool = False) -> np.ndarray:
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
def resize_tensor(image: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w, c = image.shape
    xScale = x/(w-1)
    yScale = y/(h-1)
    newImage = np.zeros([y, x, c])
    for i in range(y-1):
        for j in range(x-1):
            newImage[i + 1, j + 1] = image[1 + int(i / yScale),
                                           1 + int(j / xScale)]
    return newImage


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def get_color_fast(idx):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    CYAN = (255, 255, 0)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 165, 255)
    PURPLE = (255, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    color_pool = [RED, GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, WHITE]
    color = color_pool[idx % 8]
    return color
