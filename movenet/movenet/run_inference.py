import sys,os
sys.path.append(os.getcwd())

import numpy as np
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
import argparse

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

my_parser = argparse.ArgumentParser(description='Inferencer for MoveNet')
my_parser.add_argument('--image',
                       default=None,
                       help='Path to an image.')
my_parser.add_argument('--image_dir',
                       default="/home/chen/data/210423_ICU_raw_video_2/rgb",
                       help='Path to the directory with the images.')
my_parser.add_argument('--output_dir',
                       default="/home/chen/data/210423_ICU_raw_video_2/results",
                       help='Path to the directory to save the output.')
my_parser.add_argument('--model',
                       default="Lighning",
                       choices={"Thunder", "Lighning"},
                       help='`Thunder` for accuracy and `Lighning` for speed.')

args = my_parser.parse_args()


if __name__ == "__main__":

    if args.model == 'Lighning':
        logger.info("lightning ...")
        module = hub.load(f"{MODEL_URL}/lightning/3")
        input_size = 192
        movenet = module.signatures['serving_default']
    elif args.model == 'Thunder':
        logger.info("thunder ...")
        module = hub.load(f"{MODEL_URL}/thunder/3")
        input_size = 256
        movenet = module.signatures['serving_default']
    else:
        raise ValueError(f"Unknown model : {args.model}")

    if args.image is not None:
        assert os.path.exists(args.image)
        image_paths = [args.image]
    elif args.image_dir is not None:
        image_paths = [os.path.join(args.image_dir, i) for i in os.listdir(args.image_dir)]
        for i in image_paths:
            assert os.path.exists(i)
    else:
        raise ValueError("Please provide either folder with images or path to an image.")

    timing = []
    for image_path in image_paths:
        # Load the input image.
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)

        # Output is a [1, 1, 17, 3] tensor.
        time_now = time.time()
        keypoints_with_scores = run_inference_single(movenet, image, input_size)
        timing.append(time.time() - time_now)

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
        plt.savefig(os.path.join(args.output_dir, image_path.split('/')[-1]))
        logger.info(f"Result saved to {os.path.join(args.output_dir, image_path.split('/')[-1])}")
        plt.close()
    
    logger.info(f"Average inference time : {sum(timing[10:]) / len(timing[10:])}s")
    logger.info(f"Average FPS : {1 / (sum(timing[10:]) / len(timing[10:]))}")
