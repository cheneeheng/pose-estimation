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
