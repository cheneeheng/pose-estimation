"""
MoveNet model from tensorflow.
https://www.tensorflow.org/hub/tutorials/movenet

Code taken from:
https://github.com/tensorflow/hub/blob/master/examples/colab/movenet.ipynb

Code is adapted to run with webcam. 

Sample image taken from:
https://images.pexels.com/photos/4384679/pexels-photo-4384679.jpeg
"""

import numpy as np
from matplotlib import pyplot as plt
import time
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub

from utils.image_single import run_inference as run_inference_single
from utils.image_sequence_crop import init_crop_region, determine_crop_region
from utils.image_sequence_crop import run_inference as run_inference_sequence_crop
from utils.visualization import draw_prediction_on_image, to_gif

from common import *

logger = tf.get_logger()
logger.setLevel('INFO')
logger.propagate = False

# [DUMMY RUN] ******************************************************************
logger.info("Dummy run ...")
module = hub.load(f"{MODEL_URL}/lightning/3")
input_size = 192
movenet = module.signatures['serving_default']

# Load the input image.
image_path = 'sample/pexels-photo-4384679.jpeg'
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image)

for _ in range(100):
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = run_inference_single(movenet, image, input_size)
# ****************************************************************** [DUMMY RUN]

# [Single Image] *******************************************************
logger.info("lightning ...")
module = hub.load(f"{MODEL_URL}/lightning/3")
input_size = 192
movenet = module.signatures['serving_default']

# Load the input image.
image_path = 'sample/pexels-photo-4384679.jpeg'
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image)

time_now = time.time()

for _ in range(100):
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = run_inference_single(movenet, image, input_size)

logger.info(f"Average inference time from 100 runs : {(time.time() - time_now) / 100}s")
logger.info(f"Average FPS : {1 / ((time.time() - time_now) / 100)}")

# Visualize the predictions with image.
display_image = tf.expand_dims(image, axis=0)
display_image = tf.cast(
    tf.image.resize_with_pad(display_image, 1280, 1280), 
    dtype=tf.int32
)
output_overlay = draw_prediction_on_image(
    np.squeeze(display_image.numpy(), axis=0), 
    keypoints_with_scores
)

plt.figure(figsize=(5, 5))
plt.imshow(output_overlay)
_ = plt.axis('off')
plt.savefig('result/pexels-photo-4384679.png')
logger.info("Result saved to `result/pexels-photo-4384679.png` ...")
# ******************************************************* [Single Image]
# [Single Image] *******************************************************
logger.info("thunder ...")
module = hub.load(f"{MODEL_URL}/thunder/3")
input_size = 256
movenet = module.signatures['serving_default']

# Load the input image.
image_path = 'sample/pexels-photo-4384679.jpeg'
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image)

time_now = time.time()

for _ in range(100):
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = run_inference_single(movenet, image, input_size)

logger.info(f"Average inference time from 100 runs : {(time.time() - time_now) / 100}s")
logger.info(f"Average FPS : {1 / ((time.time() - time_now) / 100)}")

# Visualize the predictions with image.
display_image = tf.expand_dims(image, axis=0)
display_image = tf.cast(
    tf.image.resize_with_pad(display_image, 1280, 1280), 
    dtype=tf.int32
)
output_overlay = draw_prediction_on_image(
    np.squeeze(display_image.numpy(), axis=0), 
    keypoints_with_scores
)

plt.figure(figsize=(5, 5))
plt.imshow(output_overlay)
_ = plt.axis('off')
plt.savefig('result/pexels-photo-4384679.png')
logger.info("Result saved to `result/pexels-photo-4384679.png` ...")
# ******************************************************* [Single Image]

# # [Image Sequence] *****************************************************
# module = hub.load(f"{MODEL_URL}/lightning/3")
# input_size = 192
# movenet = module.signatures['serving_default']

# # Load the input image.
# image_path = 'sample/dance_input.gif'
# image = tf.io.read_file(image_path)
# image = tf.image.decode_gif(image)

# num_frames, image_height, image_width, _ = image.shape
# crop_region = init_crop_region(image_height, image_width)

# output_images = []
# for frame_idx in tqdm(range(num_frames)):
#     keypoints_with_scores = run_inference_sequence_crop(
#         movenet, 
#         image[frame_idx, :, :, :], 
#         crop_region,
#         crop_size=[input_size, input_size]
#     )
#     output_images.append(
#         draw_prediction_on_image(
#             image[frame_idx, :, :, :].numpy().astype(np.int32),
#             keypoints_with_scores, 
#             crop_region=None,
#             close_figure=True, 
#             output_image_height=300
#         )
#     )
#     crop_region = determine_crop_region(
#         keypoints_with_scores, 
#         image_height, 
#         image_width
#     )
#     time.sleep(1)

# # Prepare gif visualization.
# output = np.stack(output_images, axis=0)
# to_gif('result/dance_input.gif', output, fps=10)
# logger.info("Result saved to `result/dance_input.gif` ...")
# # ***************************************************** [Image Sequence]
# # [Image Sequence] *****************************************************
# module = hub.load(f"{MODEL_URL}/thunder/3")
# input_size = 256
# movenet = module.signatures['serving_default']

# # Load the input image.
# image_path = 'sample/dance_input.gif'
# image = tf.io.read_file(image_path)
# image = tf.image.decode_gif(image)

# num_frames, image_height, image_width, _ = image.shape
# crop_region = init_crop_region(image_height, image_width)

# output_images = []
# for frame_idx in tqdm(range(num_frames)):
#     keypoints_with_scores = run_inference_sequence_crop(
#         movenet, 
#         image[frame_idx, :, :, :], 
#         crop_region,
#         crop_size=[input_size, input_size]
#     )
#     output_images.append(
#         draw_prediction_on_image(
#             image[frame_idx, :, :, :].numpy().astype(np.int32),
#             keypoints_with_scores, 
#             crop_region=None,
#             close_figure=True, 
#             output_image_height=300
#         )
#     )
#     crop_region = determine_crop_region(
#         keypoints_with_scores, 
#         image_height, 
#         image_width
#     )
#     time.sleep(1)

# # Prepare gif visualization.
# output = np.stack(output_images, axis=0)
# to_gif('result/dance_input.gif', output, fps=10)
# logger.info("Result saved to `result/dance_input.gif` ...")
# # ***************************************************** [Image Sequence]
