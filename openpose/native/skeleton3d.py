#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 10:49:07 2021

@author: kai
"""

# =============================================================================
# Generate 3D Skeleton with Realsense Depth Map
# =============================================================================

import sys
sys.path.append('/usr/local/python/openpose')
import pyopenpose as op

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from io_utils import *

# =============================================================================
# bone_list:  {0:neck}, {1:torso}
#             {2:left_shoulder}, {3:right_shoulder}, {4:left_upper_arm}, {5:right_upper_arm}, {6:left_forearm}, {7:right_forearm}
#             {8:left_hip}, {9:right_hip}, {10:left_thigh}, {11:right_thigh}, {12:left_calf}, {13:right_calf}
# =============================================================================
bone_list = [[0,1], [1,8],
             [1,2], [1,5], [2,3], [5,6], [3,4], [6,7],
             [8,9,], [8,12], [9,10], [12,13], [10,11], [13,14]]
color_list = ['tomato', 'r',
              'g', 'orangered', 'limegreen', 'darkorange', 'lime', 'orange',
              'b', 'darkviolet','royalblue', 'fuchsia', 'cornflowerblue', 'violet']

def get_bounding_box(skeletons):
    boxes = []
    if len(skeletons) == 0:
        return boxes
    for s in skeletons:
        score_avg = np.mean(s[:,2])
        if score_avg < 0.4: 
            #boxes.append([])
            continue
        xmin = sys.maxsize
        xmax = -1
        ymin = xmin
        ymax = xmax
        for x, y, _ in s:
            if x < 1e-6 and y < 1e-6: continue
            if x > xmax: xmax = x
            if x < xmin: xmin = x
            if y > ymax: ymax = y
            if y < ymin: ymin = y
        if xmax > xmin and ymax > ymin:
            boxes.append([int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)])
        #else:
            #boxes.append([])
    return boxes


def get_overlapping_area(box1,box2):
    # calculate overlapping area of 2 rectangular ROI boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_ = x1 + w1
    x2_ = x2 + w2
    y1_ = y1 + h1
    y2_ = y2 + h2
    xs = [x1, x1_, x2, x2_]
    ys = [y1, y1_, y2, y2_]
    xs.sort()
    ys.sort()
    non_overlap = [[x1_, x2], [x2_, x1], [y1_, y2], [y2_, y1]]
    if xs[1:3]==non_overlap[0] or xs[1:3]==non_overlap[1] or ys[1:3]==non_overlap[2] or ys[1:3]==non_overlap[3]:
        return 0
    else:
        overlap_area = (xs[2]-xs[1]) * (ys[2]-ys[1])
        return overlap_area

def match_boxes_by_overlapping_area(boxes1, boxes2, idx, next_id, thres=0.6):
    # COULD HAVE PROBLEMS!!!
    # boxes1: boxes in previous frame
    # boxes2: boxes in current frame
    #match_id = []
    resorted_boxes2 = []
    for b1 in boxes1:
        found = 0
        for b2 in boxes2:
           overlap = get_overlapping_area(b1, b2) 
           if overlap / b1[2] / b1[3] > thres:
               #match_id.append(boxes2.index(b2))
               resorted_boxes2.append(b2)
               boxes2.remove(b2)
               found = 1
               break
        if found == 0:
            idx.remove(idx[boxes1.index(b1)])
    # for i in match_id:
    #     if boxes2[i] in boxes2:
    #         resorted_boxes2.append(boxes2[i])
    #         boxes2.remove(boxes2[i])
    for b in boxes2:
        resorted_boxes2.append(b)
        idx.append(next_id)
        next_id += 1
    return [resorted_boxes2, idx, next_id]


def get_3d_skeletons(skeletons, depth_img, intr_mat):
    H, W = depth_img.shape
    fx = intr_mat[0,0]
    fy = intr_mat[1,1]
    cx = intr_mat[0,2]
    cy = intr_mat[1,2]
    skeletons3d = []
    for s in skeletons:
        joints3d = []
        score_avg = np.mean(s[:,2])
        if score_avg < 0.4: 
            #skeletons3d.append([])
            continue
        for x, y, _ in s:
            patch = depth_img[max(0,int(y-2)):min(H, int(y+2)), max(0,int(x-2)):min(W,int(x+2))]
            depth_avg = np.mean(patch)
            x3d = (x-cx) / fx * depth_avg
            y3d = (y-cy) / fy * depth_avg
            joints3d.append([x3d, y3d, depth_avg])
        skeletons3d.append(joints3d)
    return np.array(skeletons3d)

def height_estimation(skeletons):
    # check if necessary joints are visible: (0), 1, 8; 9, 10, 11; 12, 13, 14
    heights = []
    for s in skeletons:
        torso = -1
        leg1 = -1
        leg2 = -1
        if np.count_nonzero(s[0]) > 0 and np.count_nonzero(s[1]) > 0 and np.count_nonzero(s[8]) > 0:
            torso = np.linalg.norm(s[0]-s[1]) + np.linalg.norm(s[1]-s[8])
        if np.count_nonzero(s[9]) > 0 and np.count_nonzero(s[10]) > 0 and np.count_nonzero(s[11]) > 0:
            leg1 = np.linalg.norm(s[9]-s[10]) + np.linalg.norm(s[10]-s[11])
        if np.count_nonzero(s[12]) > 0 and np.count_nonzero(s[13]) > 0 and np.count_nonzero(s[14]) > 0:
            leg2 = np.linalg.norm(s[12]-s[13]) + np.linalg.norm(s[13]-s[14])
        if torso == -1:
            heights.append(-1)
        else:
            if leg1 == -1 and leg2 == -1:
                heights.append(-1)
            elif leg1 != -1 and leg2 != -1:
                heights.append(torso + (leg1+leg2)/2 )
            elif leg1 != -1:
                heights.append(torso + leg1)
            else:
                heights.append(torso + leg2)
                
    return np.array(heights)

def get_valid_limbs(skeletons, mask_joint, bone_list=bone_list):
    mask_limb = []
    for b in bone_list:
        if mask_joint[b[0]]==False or mask_joint[b[1]]==False:
            mask_limb.append(False)
        else:
            mask_limb.append(True)
    return mask_limb
    

def get_limb_lengths(skeletons, bone_list=bone_list):
    feat_limb = []
    for s in skeletons:
        limb_length = []
        mask_joint = s[:15].any(axis=1)
        mask_limb = get_valid_limbs(skeletons, mask_joint, bone_list)
        for b, m in zip(bone_list, mask_limb):
            if m:
                limb_len = np.linalg.norm(s[b[0]]-s[b[1]])
                limb_length.append(limb_len)
            else:
                limb_length.append(-1)
        feat_limb.append(limb_length)
    return feat_limb
            


def draw_skeletons(skeletons3d, bones=bone_list, colors=color_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for s in skeletons3d:
        mask = s[:15].any(axis=1)
        s_nonzero = s[:15][mask]
        ax.scatter(s_nonzero[:,0], s_nonzero[:, 1], s_nonzero[:, 2], marker='o')
        for b, c in zip(bones, colors):
            if mask[b[0]]==True and mask[b[1]]==True:
                ax.plot([s[b[0]][0], s[b[1]][0]], [s[b[0]][1], s[b[1]][1]], zs=[s[b[0]][2], s[b[1]][2]], c=c)
    ax.axis(xmin=-2000, xmax=2000, ymin=-2000, ymax=2000)
    ax.set_zlim(500,4500)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    return 1




data_num = 1611868923.2967753

depth_folder = "/home/kai/workspace/DHM-ICU/biometricID/data/realSense/"+str(data_num)+"/depth/"
rgb_folder = "/home/kai/workspace/DHM-ICU/biometricID/data/realSense/"+str(data_num)+"/rgb/"
calib_file = "/home/kai/workspace/DHM-ICU/biometricID/data/realSense/"+str(data_num)+"/calib.txt"
timestamp_file = "/home/kai/workspace/DHM-ICU/biometricID/data/realSense/"+str(data_num)+"/timestamps.txt"

calib_data = read_realsense_config(calib_file)

timestamp_file =  open(timestamp_file, 'r') 
timestamps = timestamp_file.readlines()
timestamps = [int(ts) for ts in timestamps]
timestamp_file.close()

# Start Openpose Pipeline
params = dict()
params["model_folder"] = "/home/kai/workspace/openpose/models"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


bbox_hist = {}
try:
    boxes_pre = []
    idx = []
    next_id = 0
    for ts in timestamps:
        depth_img = np.load(depth_folder+str(ts)+'.npy')
        rgb_img = cv.imread(rgb_folder+str(ts)+'.png')
        
        # Process Image
        datum = op.Datum()
        datum.cvInputData = rgb_img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        skeletons = datum.poseKeypoints
        output_img = datum.cvOutputData
        # Bounding Box
        # get_keypoints_rectangle(keypoints_array, number_of_keypoints, thresholdRectangle, firstIndex = 0,
        #     lastIndex = -1)
        boxes = []
        if skeletons is not None:
            boxes = get_bounding_box(skeletons[:,:15,:])
        
        bbox_hist[ts] = []
        for b in boxes:
            bbox_hist[ts].append(b)
        
        if len(boxes_pre) == 0:
            for b in boxes:
                idx.append(next_id)
                next_id += 1
        else:
            boxes, idx, next_id = match_boxes_by_overlapping_area(boxes_pre, boxes, idx, next_id, 0.4)
            
        if skeletons is not None:
            skeletons3d = get_3d_skeletons(skeletons[:,:15,:], depth_img, calib_data.rgb_intrinsics)
            heights = height_estimation(skeletons3d) / 1000.0
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            cv.rectangle(output_img, (x, y), (x+w, y+h), (255,0,0), 2)
            cv.putText(output_img, 'ID: %d'%idx[i], (x,y-25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            # cv.putText(output_img, 'height: %.2f'%heights[i], (x,y-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    
        
    
        boxes_pre = boxes
            
        # Display Image
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        cv.namedWindow("People Detection", cv.WINDOW_AUTOSIZE)
        cv.imshow("People Detection", output_img)
        key = cv.waitKey(30)
        
        if key & 0xFF == ord('q') or key == 27:
            cv.destroyAllWindows()
            for i in range(5):
                cv.waitKey(1)
            break
finally:
    opWrapper.stop()
    save_bbox_path = "/home/kai/workspace/DHM-ICU/biometricID/data/realSense/"+str(data_num)+"/bbox_history.npy"
    np.save(save_bbox_path, bbox_hist)
    # read_dictionary = np.load('my_file.npy',allow_pickle='TRUE').item()
    # print(read_dictionary['hello']) # displays "world"





# =============================================================================
# ts = 388427676
# depth_img = np.load(depth_folder+str(ts)+'.npy')
# rgb_img = cv.imread(rgb_folder+str(ts)+'.png')
# 
# # Process Image
# datum = op.Datum()
# imageToProcess = rgb_img
# datum.cvInputData = imageToProcess
# opWrapper.emplaceAndPop(op.VectorDatum([datum]))
# 
# output_img = datum.cvOutputData
# skeletons = datum.poseKeypoints
# # boxes =[]
# # for j in joints:
# #     box = op.get_keypoints_rectangle(j, 0, 0.1, 0, -1)
# #     x = int(box.x)
# #     y = int(box.y)
# #     w = int(box.width)
# #     h = int(box.height)
# #     boxes.append([x, y, w, h])
# #     cv.rectangle(output_img, (x, y), (x+w, y+h), (255,0,0), 2)
# boxes = get_bounding_box(skeletons)
# skeletons3d = get_3d_skeletons(skeletons, depth_img, calib_data.rgb_intrinsics)
# skeletons3d = np.array(skeletons3d)
# heights = height_estimation(skeletons3d) / 1000.0
# for i in range(len(boxes)):
#     x, y, w, h = boxes[i]
#     cv.rectangle(output_img, (x, y), (x+w, y+h), (255,0,0), 2)
#     cv.putText(output_img, 'ID: %d'%i, (x,y-25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
#     cv.putText(output_img, 'height: %.2f'%heights[i], (x,y-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
# 
# 
# # Display Image
# # print("Body keypoints: \n" + str(datum.poseKeypoints))
# plt.figure()
# # plt.imshow(datum.cvOutputData)
# plt.imshow(output_img)
# draw_skeletons(skeletons3d)
# 
# feat_limb = get_limb_lengths(skeletons3d, bone_list)
# 
# =============================================================================


















