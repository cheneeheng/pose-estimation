import argparse
from .utils import str2bool


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Extract skeleton using OPENPOSE')

    # OPEN POSE OPTIONS --------------------------------------------------------
    p.add_argument('--op-model-folder',
                   type=str,
                   default="/usr/local/src/openpose/models/",
                   help='folder with trained openpose models.')
    p.add_argument('--op-model-pose',
                   type=str,
                   default="BODY_25",
                   help='pose model name')
    p.add_argument('--op-net-resolution',
                   type=str,
                   default="-1x368",
                   help='resolution of input to openpose.')
    p.add_argument('--op-skel-thres',
                   type=float,
                   default=0.5,
                   help='threshold for valid skeleton.')
    p.add_argument('--op-max-true-body',
                   type=int,
                   default=8,
                   help='max number of skeletons to save.')
    p.add_argument('--op-heatmaps-add-parts',
                   type=str2bool,
                   default=True,
                   help='')
    p.add_argument('--op-heatmaps-add-bkg',
                   type=str2bool,
                   default=True,
                   help='')
    p.add_argument('--op-heatmaps-add-PAFs',
                   type=str2bool,
                   default=True,
                   help='')
    p.add_argument('--op-heatmaps-scale',
                   type=int,
                   default=1,
                   help='')
    p.add_argument('--op-save-skel',
                   type=str2bool,
                   default=False,
                   help='if true, saves the 2d skeleton.')
    p.add_argument('--op-save-skel-image',
                   type=str2bool,
                   default=False,
                   help='if true, saves the 2d skeleton image.')
    p.add_argument('--op-color-image',
                   type=str,
                   default="",
                   help='path to input color image.')
    p.add_argument('--op-skel-file',
                   type=str,
                   default="skel.txt",
                   help='file to save skeleton results.')

    # DEPTH OPTIONS ------------------------------------------------------------
    p.add_argument('--op-patch-offset',
                   type=int,
                   default=2,
                   help='offset of patch used to determine depth')
    p.add_argument('--op-ntu-format',
                   type=str2bool,
                   default=False,
                   help='whether to use coordinate system of NTU')

    # DISPLAY OPTIONS ----------------------------------------------------------
    p.add_argument('--op-display',
                   type=float,
                   default=1.0,
                   help='scale for displaying skel images.')
    p.add_argument('--op-display-depth',
                   type=int,
                   default=0,
                   help='scale for displaying skel images with depth.')

    # REALSENSE OPTIONS --------------------------------------------------------
    p.add_argument('--op-rs-dir',
                   type=str,
                   default='data/mot17',
                   help='path to folder with saved rs data.')
    p.add_argument('--op-rs-image-width',
                   type=int,
                   default=1920,
                   help='image width in px')
    p.add_argument('--op-rs-image-height',
                   type=int,
                   default=1080,
                   help='image height in px')
    p.add_argument('--op-rs-extract-3d-skel',
                   type=str2bool,
                   default=False,
                   help='if true, tries to extract 3d skeleton.')
    p.add_argument('--op-rs-save-3d-skel',
                   type=str2bool,
                   default=False,
                   help='if true, saves the 3d skeleton.')
    # p.add_argument('--op-rs-save-heatmaps',
    #                type=str2bool,
    #                default=False,
    #                help='if true, saves the heatmaps/output of network.')
    p.add_argument('--op-rs-delete-image',
                   type=str2bool,
                   default=False,
                   help='if true, deletes the rs image used in inference.')

    # TRACKING OPTIONS ---------------------------------------------------------
    p.add_argument('--op-track-deepsort',
                   type=str2bool,
                   default=False,
                   help='If true performs deepsort tracking.')
    p.add_argument('--op-track-bytetrack',
                   type=str2bool,
                   default=False,
                   help='If true performs ByteTrack tracking.')
    p.add_argument('--op-track-ocsort',
                   type=str2bool,
                   default=False,
                   help='If true performs OC Sort tracking.')
    p.add_argument('--op-track-strongsort',
                   type=str2bool,
                   default=False,
                   help='If true performs StrongSort tracking.')
    p.add_argument('--op-track-buffer',
                   type=int,
                   default=30,
                   help="the frames for keep lost tracks.")

    # EXPERIMENTS OPTIONS ------------------------------------------------------
    p.add_argument('--op-save-result-image',
                   type=str2bool,
                   default=True,
                   help='If true saves tracking+pose results in image form.')
    p.add_argument('--op-proc',
                   type=str,
                   default='',
                   help='multi (mp) or single processing (sp) or none. Based on Alphapose')  # noqa

    return p
