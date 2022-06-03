"""
Code based on:
https://learnopencv.com/opencv-dnn-with-gpu-support/
https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py
"""

import os
import cv2
import time
import numpy as np

from typing import Optional, Tuple, List

from openpose.opencv.common import *
from openpose.opencv.utils import getKeypoints
from openpose.opencv.utils import getPersonwiseKeypoints
from openpose.opencv.utils import getValidPairs


class PyOpenPoseOpenCV:
    """Class to run openpose using opencv. """

    def __init__(self,
                 image_width: int = 368,
                 image_height: int = 368,
                 body_parts: Optional[dict] = None,
                 pose_pairs: Optional[list] = None,
                 conf_thres: float = 0.1) -> None:
        super().__init__()
        self.net = None
        self.image_width = image_width
        self.image_height = image_height
        self.original_image_width = None
        self.original_image_height = None
        self.body_parts = body_parts  # name: id
        self.pose_pairs = pose_pairs  # name : name
        self.conf_thres = conf_thres

    def load(self,
             proto_file: str,
             weights_file: str,
             cuda: bool = True) -> None:
        """Loads the trained model from caffe.

        Args:
            proto_file (str): Model config.
            weights_file (str): Model weights.
            cuda (bool, optional): Whether to use GPU. Defaults to True.
        """
        self.net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
        if cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Runs inference on the input image.

        Args:
            image (np.ndarray): An rgb image.

        Returns:
            np.ndarray: Joints heatmap in the form N,V,H,W.
        """
        self.original_image_width = image.shape[1]
        self.original_image_height = image.shape[0]
        inpBlob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255,
            size=(self.image_width, self.image_height),
            mean=(0, 0, 0),
            swapRB=False,
            crop=False
        )
        self.net.setInput(inpBlob)
        return self.net.forward()

    def postprocess_single(self, prediction: np.ndarray) -> list:
        """Processes the raw prediction to find 1 pose ONLY.

        Args:
            prediction (np.ndarray): Raw heatmap prediction.

        Returns:
            list: keypoints of a pose.
        """
        assert(len(self.body_parts) <= prediction.shape[1])
        points = []
        for i in range(len(self.body_parts)):
            # Slice heatmap of corresponding body's part.
            heatMap = prediction[0, i, :, :]
            # Originally, we try to find all the local maximums.
            # To simplify a sample we just find a global one.
            # However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (self.original_image_width * point[0]) / prediction.shape[3]
            y = (self.original_image_height * point[1]) / prediction.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > self.conf_thres else None)
        return points

    def postprocess_multi(self, prediction: np.ndarray
                          ) -> Tuple[np.ndarray, list]:
        """Processes the raw prediction to find N poses.

        Args:
            prediction (np.ndarray): Raw heatmap prediction.

        Returns:
            Tuple[np.ndarray, list]: List of all keypoints found,
                matching of keypoint index to unique N persons.
        """

        assert(len(self.body_parts) <= prediction.shape[1])

        # unique id for the keypoints/joints
        keypoint_id = 0
        # the keypoints
        keypoints_list = np.zeros((0, 3))
        # list of keypoints together with unique id
        detected_keypoints = []

        # get list of all the detected keypoints.
        for i in range(len(self.body_parts) - 1):
            heatMap = prediction[0, i, :, :]
            heatMap = cv2.resize(heatMap, (self.original_image_width,
                                           self.original_image_height))
            keypoints = getKeypoints(heatMap, self.conf_thres)
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1
            detected_keypoints.append(keypoints_with_id)

        # get the pairs between the joints.
        # list of np.array of (joint1, joint2, score)
        valid_pairs, invalid_pairs = getValidPairs(prediction,
                                                   detected_keypoints,
                                                   self.pose_pairs,
                                                   self.body_parts,
                                                   self.original_image_width,
                                                   self.original_image_height)

        # get the list of keypoint idx associated to a person.
        # The idx maps to the keypoints_list.
        # list of np array (joint idx that the idx position is connected to)
        personwise_keypoints = getPersonwiseKeypoints(valid_pairs,
                                                      invalid_pairs,
                                                      keypoints_list,
                                                      self.body_parts,
                                                      self.pose_pairs)

        return keypoints_list, personwise_keypoints

    def draw_skeleton(self,
                      points: Optional[list] = None,
                      personwise_keypoints: Optional[List[np.ndarray]] = None,
                      keypoints_list: Optional[np.ndarray] = None,
                      image: Optional[np.ndarray] = None,
                      show_image: bool = False,
                      output_image_path: Optional[str] = None):

        if image is None:
            return None

        # single pose ----------------------------------------------------------
        if points is not None:

            for pair in self.pose_pairs:
                part_from = pair[0]
                part_to = pair[1]
                assert(part_from in self.body_parts)
                assert(part_to in self.body_parts)

                id_from = self.body_parts[part_from]
                id_to = self.body_parts[part_to]

                if points[id_from] and points[id_to]:
                    cv2.line(image, points[id_from],
                             points[id_to], (0, 255, 0), 3)
                    cv2.ellipse(image, points[id_from], (3, 3),
                                0, 0, 360, (0, 0, 255), cv2.FILLED)
                    cv2.ellipse(image, points[id_to], (3, 3),
                                0, 0, 360, (0, 0, 255), cv2.FILLED)

        # multiple poses -------------------------------------------------------
        elif personwise_keypoints is not None and keypoints_list is not None:

            # KEYPOINT VISUALIZATION
            # for i in range(len(self.body_parts) - 1):
            #     for j in range(len(detected_keypoints[i])):
            #         cv2.circle(image,
            #                    detected_keypoints[i][j][0:2],
            #                    5, (255, 105, 180), -1, cv2.LINE_AA)
            for i in range(keypoints_list.shape[0]):
                cv2.circle(image,
                           keypoints_list[i][0:2].tolist(),
                           5, (255, 105, 180), -1, cv2.LINE_AA)

            # KEYPOINT PAIRS VISUALIZATION
            for idx, pose_pair in enumerate(self.pose_pairs):
                for single_person_keypoints in personwise_keypoints:
                    index = single_person_keypoints[np.array(
                        [self.body_parts[x] for x in pose_pair])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(image,
                             (B[0], A[0]),
                             (B[1], A[1]),
                             COLORS[idx], 3, cv2.LINE_AA)

        t, _ = self.net.getPerfProfile()
        freq = cv2.getTickFrequency() / 1000
        cv2.putText(image, '%.2fms' % (t / freq), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        if output_image_path is not None:
            assert isinstance(output_image_path, str)
            assert os.path.exists(output_image_path)
            cv2.imwrite(output_image_path, image)

        if show_image:
            cv2.imshow('OpenPose using OpenCV', image)
            cv2.waitKey(0)
