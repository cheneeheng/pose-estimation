"""
Code based on:
https://learnopencv.com/opencv-dnn-with-gpu-support/
https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py
"""

import os
import cv2
import time
import numpy as np

# for COCO
COLORS = [
    [0, 100, 255],
    [0, 100, 255],
    [0, 255, 255],
    [0, 100, 255],
    [0, 255, 255],
    [0, 100, 255],
    [0, 255, 0],
    [255, 200, 100],
    [255, 0, 255],
    [0, 255, 0],
    [255, 200, 100],
    [255, 0, 255],
    [0, 0, 255],
    [255, 0, 0],
    [200, 200, 0],
    [255, 0, 0],
    [200, 200, 0],
    [0, 0, 0]
]
# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output,
# Similarly, (1,5) -> (39,40) and so on.
MAPIDX = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
          [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
          [47, 48], [49, 50], [53, 54], [51, 52], ]


def get_body_parts_and_pose_pairs(dataset=None):
    if dataset == 'COCO':
        body_parts = {"Nose": 0,
                      "Neck": 1,
                      "RShoulder": 2,
                      "RElbow": 3,
                      "RWrist": 4,
                      "LShoulder": 5,
                      "LElbow": 6,
                      "LWrist": 7,
                      "RHip": 8,
                      "RKnee": 9,
                      "RAnkle": 10,
                      "LHip": 11,
                      "LKnee": 12,
                      "LAnkle": 13,
                      "REye": 14,
                      "LEye": 15,
                      "REar": 16,
                      "LEar": 17,
                      "Background": 18}
        pose_pairs = [["Neck", "RShoulder"],
                      ["Neck", "LShoulder"],
                      ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"],
                      ["LShoulder", "LElbow"],
                      ["LElbow", "LWrist"],
                      ["Neck", "RHip"],
                      ["RHip", "RKnee"],
                      ["RKnee", "RAnkle"],
                      ["Neck", "LHip"],
                      ["LHip", "LKnee"],
                      ["LKnee", "LAnkle"],
                      ["Neck", "Nose"],
                      ["Nose", "REye"],
                      ["REye", "REar"],
                      ["Nose", "LEye"],
                      ["LEye", "LEar"]]
    elif dataset == 'MPI':
        body_parts = {"Head": 0,
                      "Neck": 1,
                      "RShoulder": 2,
                      "RElbow": 3,
                      "RWrist": 4,
                      "LShoulder": 5,
                      "LElbow": 6,
                      "LWrist": 7,
                      "RHip": 8,
                      "RKnee": 9,
                      "RAnkle": 10,
                      "LHip": 11,
                      "LKnee": 12,
                      "LAnkle": 13,
                      "Chest": 14,
                      "Background": 15}
        pose_pairs = [["Head", "Neck"],
                      ["Neck", "RShoulder"],
                      ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"],
                      ["Neck", "LShoulder"],
                      ["LShoulder", "LElbow"],
                      ["LElbow", "LWrist"],
                      ["Neck", "Chest"],
                      ["Chest", "RHip"],
                      ["RHip", "RKnee"],
                      ["RKnee", "RAnkle"],
                      ["Chest", "LHip"],
                      ["LHip", "LKnee"],
                      ["LKnee", "LAnkle"]]
    else:
        raise(Exception("you need to specify either 'COCO', 'MPI' in dataset"))
    return body_parts, pose_pairs


def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []
    # find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(prediction,
                  detected_keypoints,
                  pose_pairs,
                  body_parts,
                  image_width,
                  image_height):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.5

    # loop for every POSE_PAIR
    for k in range(len(MAPIDX)):
        # A->B constitute a limb
        pafA = prediction[0, MAPIDX[k][0], :, :]
        pafB = prediction[0, MAPIDX[k][1], :, :]
        pafA = cv2.resize(pafA, (image_width, image_height))
        pafB = cv2.resize(pafB, (image_width, image_height))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[body_parts[pose_pairs[k][0]]]
        candB = detected_keypoints[body_parts[pose_pairs[k][1]]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between
        # the joints
        # Use the above formula to compute a score to mark
        # the connection valid

        if(nA != 0 and nB != 0):
            valid_pair = np.zeros((0, 3))

            for i in range(nA):
                max_j = -1
                maxScore = -1
                found = 0

                for j in range(nB):

                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue

                    # Find p(u)
                    interp_coord = list(
                        zip(np.linspace(candA[i][0], candB[j][0],
                                        num=n_interp_samples),
                            np.linspace(candA[i][1], candB[j][1],
                                        num=n_interp_samples)))

                    # Find L(p(u))
                    paf_interp = []
                    for x in range(len(interp_coord)):
                        paf_interp.append(
                            [pafA[int(round(interp_coord[x][1])),
                                  int(round(interp_coord[x][0]))],
                             pafB[int(round(interp_coord[x][1])),
                                  int(round(interp_coord[x][0]))]])

                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF
                    # is higher then threshold -> Valid Pair
                    if len(np.where(paf_scores > paf_score_th)[0]) \
                            / n_interp_samples > conf_th:
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1

                # Append the connection to the list
                if found:
                    valid_pair = np.append(
                        valid_pair,
                        [[candA[i][3], candB[max_j][3], maxScore]],
                        axis=0
                    )

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)

        else:  # If no keypoints are detected
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])

    return valid_pairs, invalid_pairs


# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs,
                           invalid_pairs,
                           keypoints_list,
                           body_parts,
                           pose_pairs):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(MAPIDX)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:, 0]
            partBs = valid_pairs[k][:, 1]
            indexA, indexB = np.array([body_parts[x] for x in pose_pairs[k]])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]  # noqa

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints
                    # and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]  # noqa
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])

    return personwiseKeypoints


class PyOpenPose(object):
    def __init__(self,
                 image_width: int = 368,
                 image_height: int = 368,
                 body_parts: dict = None,
                 pose_pairs: list = None,
                 conf_thres: float = 0.1) -> None:
        super().__init__()
        self.net = None
        self.image_width = image_width
        self.image_height = image_height
        self.original_image_width = None
        self.original_image_height = None
        self.body_parts = body_parts
        self.pose_pairs = pose_pairs
        self.conf_thres = conf_thres

    def load(self, proto_file, weights_file, cuda: bool = True):
        self.net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
        if cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def predict(self, image):
        self.original_image_width = image.shape[1]
        self.original_image_height = image.shape[0]
        inpBlob = cv2.dnn.blobFromImage(image,
                                        scalefactor=1.0 / 255,
                                        size=(self.image_width,
                                              self.image_height),
                                        mean=(0, 0, 0),
                                        swapRB=False,
                                        crop=False)
        self.net.setInput(inpBlob)
        prediction = self.net.forward()
        return prediction

    def postprocess_single(self,
                           prediction,
                           image=None,
                           show_image=False,
                           output_image_path=None):

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

        if image is not None:

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

        return points, image

    def postprocess_multi(self,
                          prediction,
                          image=None,
                          show_image=False,
                          output_image_path=None):

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

        if image is not None:

            # KEYPOINT VISUALIZATION
            for i in range(len(self.body_parts) - 1):
                for j in range(len(detected_keypoints[i])):
                    cv2.circle(image,
                               detected_keypoints[i][j][0:2],
                               5, (255, 105, 180), -1, cv2.LINE_AA)

            # get the pairs between the joints.
            valid_pairs, invalid_pairs = getValidPairs(prediction,
                                                       detected_keypoints,
                                                       self.pose_pairs,
                                                       self.body_parts,
                                                       self.image_width,
                                                       self.image_height)

            # get the list of keypoint idx associated to a person.
            # The idx maps to the keypoints_list.
            personwiseKeypoints = getPersonwiseKeypoints(valid_pairs,
                                                         invalid_pairs,
                                                         keypoints_list,
                                                         self.body_parts,
                                                         self.pose_pairs)

            # KEYPOINT PAIRS VISUALIZATION
            for i in range(len(self.pose_pairs)):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(
                        [self.body_parts[x] for x in self.pose_pairs[i]])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(image,
                             (B[0], A[0]),
                             (B[1], A[1]),
                             COLORS[i], 3, cv2.LINE_AA)

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

        return keypoints_list, personwiseKeypoints, image


if __name__ == "__main__":

    image_path = "opencv/sandbox/pexels-photo-4384679.jpeg"
    image = cv2.imread(image_path)
    image = cv2.resize(image, (368, 368))
    # cv2.imshow('OpenPose using OpenCV', image)
    # cv2.waitKey(0)

    basepath = "/home/chen/data/03_OpenPose/models"
    protoFile = basepath + "/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = basepath + "/pose/coco/pose_iter_440000.caffemodel"
    dataset = 'COCO'

    # basepath = "/home/chen/data/03_OpenPose/models"
    # protoFile = basepath + "/pose/mpi/pose_deploy_linevec.prototxt"
    # weightsFile = basepath + "/pose/mpi/pose_iter_160000.caffemodel"
    # dataset = 'MPI'

    body_parts, pose_pairs = get_body_parts_and_pose_pairs(dataset)

    op = PyOpenPose(body_parts=body_parts, pose_pairs=pose_pairs)
    op.load(protoFile, weightsFile)

    image_height = 368*1
    image_width = 368*1

    for _ in range(10):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_width, image_height))
        op.image_height = image_height
        op.image_width = image_width
        pred = op.predict(image)
        # op.postprocess_single(pred, image,
        #                       output_image_path="opencv/output/tmp.png")
        op.postprocess_multi(pred, image,
                             output_image_path="opencv/output/tmp.png")
