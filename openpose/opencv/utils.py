import cv2
import numpy as np

from typing import Tuple, List

from openpose.opencv.common import MAPIDX


def getKeypoints(probMap: np.ndarray, threshold: float = 0.1) -> list:
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
        _, _, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(prediction: np.ndarray,
                  detected_keypoints: list,
                  pose_pairs: dict,
                  body_parts: dict,
                  image_width: int,
                  image_height: int) -> Tuple[list, list]:
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
            # valid_pair = np.zeros((0, 3))
            valid_pair = []

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
                    # valid_pair = np.append(
                    #     valid_pair,
                    #     [[candA[i][3], candB[max_j][3], maxScore]],
                    #     axis=0
                    # )
                    valid_pair.append([candA[i][3], candB[max_j][3], maxScore])

            # Append the detected connections to the global list
            if len(valid_pair) > 0:
                valid_pair = np.array(valid_pair)
            else:
                valid_pair = np.zeros((0, 3))
            valid_pairs.append(valid_pair)

        else:  # If no keypoints are detected
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])

    return valid_pairs, invalid_pairs


# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs: list,
                           invalid_pairs: list,
                           keypoints_list: np.ndarray,
                           body_parts: dict,
                           pose_pairs: dict) -> List[np.ndarray]:
    # the last number in each row is the overall score
    personwise_keypoints = []

    for k in range(len(MAPIDX)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:, 0]
            partBs = valid_pairs[k][:, 1]
            indexA, indexB = np.array([body_parts[x] for x in pose_pairs[k]])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j, keypoints_j in enumerate(personwise_keypoints):
                    if keypoints_j[indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwise_keypoints[person_idx][indexB] = partBs[i]
                    personwise_keypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]  # noqa

                # if find no partA in the subset, create a new subset
                elif not found and k < len(body_parts)-2:
                    row = -1 * np.ones(len(body_parts))
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints
                    # and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]  # noqa
                    personwise_keypoints.append(row)

    return personwise_keypoints
