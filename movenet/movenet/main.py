"""
MoveNet model from tensorflow.
https://www.tensorflow.org/hub/tutorials/movenet

Code taken from:
https://github.com/tensorflow/hub/blob/master/examples/colab/movenet.ipynb

Code is adapted to run with webcam. 

Sample image taken from:
https://images.pexels.com/photos/4384679/pexels-photo-4384679.jpeg
"""

import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# import time
# from tqdm import tqdm

# import tensorflow as tf
# import tensorflow_hub as hub

# from movenet.image_single import run_inference as run_inference_single
# from movenet.image_sequence_crop import init_crop_region, determine_crop_region
# from movenet.image_sequence_crop import run_inference as run_inference_sequence_crop
# from movenet.visualization import draw_prediction_on_image, to_gif

from common import *


if __name__ == "__main__":
    img = cv2.imread('sample/pexels-photo-4384679.jpeg')
    img = cv2.resize(img, (640, 640))
    cv2.imshow("img", img)
    cv2.waitKey( )
    exit(1)

    # print(cv2.getBuildInformation())

    # cap = cv2.VideoCapture('udp://127.0.0.1:5000',cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
