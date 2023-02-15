import argparse
import numpy as np


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
