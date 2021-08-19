"""
Code taken from:
https://github.com/tensorflow/hub/blob/master/examples/colab/movenet.ipynb
"""
import tensorflow as tf

from common import *


def run_inference(movenet, image, input_size):
    """Runs model inferece on an image.

    The function runs the model inference on the cropped region and updates the
    model output to the original image coordinate system.
    """
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.cast(tf.image.resize_with_pad(
        input_image, input_size, input_size), dtype=tf.int32)
    # Run model inference.
    outputs = movenet(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0']
    return keypoints_with_scores
