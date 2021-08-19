"""
Code taken from:
https://learnopencv.com/opencv-dnn-with-gpu-support/
"""

import cv2
import time

# print(cv2.getBuildInformation())

# ------------------------------------------------------------------------------
# Read image
# frame = cv2.imread("opencv/pexels-photo-4384679.jpeg")
frame = cv2.imread("examples/media/COCO_val2014_000000000192.jpg")
# frame = cv2.imread("examples/media/COCO_val2014_000000000241.jpg")
frame = cv2.resize(frame, (368, 368))
cv2.imshow("Output-Keypoints", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------------------------------------------------------
# Path to models in the openpose repo with the download script.
basepath = "/home/chen/data/03_OpenPose/models"

# Specify the paths for the 2 files
protoFile = basepath + "/pose/coco/pose_deploy_linevec.prototxt"
weightsFile = basepath + "/pose/coco/pose_iter_440000.caffemodel"

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Cuda version
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Specify the input image dimensions
inWidth = 368
inHeight = 368

# Prepare the frame to be fed to the network
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)

# Set the prepared object as the input blob of the network
net.setInput(inpBlob)

# Inference.
output = net.forward()

# ------------------------------------------------------------------------------
H = output.shape[2]
W = output.shape[3]
# Empty list to store the detected keypoints
points = []
for i in range(15):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # Scale the point to fit on the original image
    x = (inWidth * point[0]) / W
    y = (inHeight * point[1]) / H

    threshold = 0.2
    if prob > threshold:
        cv2.circle(frame,
                   (int(x), int(y)),
                   3,
                   (0, 255, 255),
                   thickness=-1,
                   lineType=cv2.FILLED)  # noqa
        cv2.putText(frame,
                    "{}".format(i),
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    lineType=cv2.LINE_AA)

        # Add the point to the list if the probability
        # is greater than the threshold
        points.append((int(x), int(y)))
    else:
        points.append(None)

# ------------------------------------------------------------------------------
i = 0
probMap = output[0, i, :, :]
probMap = cv2.resize(probMap, (inWidth, inHeight)) * 255
cv2.imwrite("probMap.png", probMap.astype(int))

# ------------------------------------------------------------------------------
cv2.imwrite("tmp.png", frame)
cv2.imshow("Output-Keypoints", frame)
cv2.waitKey(0)
# cv2.destroyAllWindows()

# ------------------------------------------------------------------------------
# # Draw skeletons
# for pair in POSE_PAIRS:
#     partA = pair[0]
#     partB = pair[1]

#     if points[partA] and points[partB]:
#         cv2.line(frame, points[partA], points[partB], (0, 255, 0), 3)
