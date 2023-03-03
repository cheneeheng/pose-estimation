# TAKEN FROM AlphaPose/scripts/demo_inference.py

import argparse

"""----------------------------- Demo options -------------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg',
                    type=str,
                    required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint',
                    type=str,
                    required=True,
                    help='checkpoint file name')
parser.add_argument('--sp',
                    default=False,
                    action='store_true',
                    help='Use single process for pytorch')  # see paper Fig4.
parser.add_argument('--detector', dest='detector',
                    help='detector name',
                    default="yolo")  # AlphaPose/detector/apis.py
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory',
                    default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list',
                    default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name',
                    default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory',
                    default="examples/res/")
parser.add_argument('--save_img',
                    default=False,
                    action='store_true',
                    help='save result as image')
parser.add_argument('--vis',
                    default=False,
                    action='store_true',
                    help='visualize image')
parser.add_argument('--final_img_scale',
                    type=float,
                    default=1.0,
                    help='scale of visualized/saved image')
parser.add_argument('--showbox',
                    default=True,
                    action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile',
                    default=False,
                    action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')  # noqa
parser.add_argument('--min_box_area',
                    type=int,
                    default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch',
                    type=int,
                    default=1,  # 5
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch',
                    type=int,
                    default=1,  # 64
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--gpus', dest='gpus',
                    type=str,
                    default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')  # noqa
parser.add_argument('--debug',
                    default=False,
                    action='store_true',
                    help='print detail information')
# KEEP DEFAULT ------
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file',
                    default="")
parser.add_argument('--qsize', dest='qsize',
                    type=int,
                    default=16,  # 1024
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')  # noqa
parser.add_argument('--flip',
                    default=False,
                    action='store_true',
                    help='enable flip testing')
parser.add_argument('--eval', dest='eval',
                    default=False,
                    action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')  # noqa

"""----------------------------- Video options ------------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')  # noqa
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)  # noqa
"""----------------------------- Tracking options ---------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)  # noqa
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)  # noqa
parser.add_argument('--pose_track_model', dest='pose_track_model',
                    help='reid model', default='')  # noqa


args = parser.parse_args()
