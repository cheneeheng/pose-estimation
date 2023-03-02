# TAKEN FROM AlphaPose/scripts/demo_inference.py

"""Script for single-gpu/multi-gpu demo."""

import os
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from submodules.AlphaPose.detector.apis import get_detector
from submodules.AlphaPose.trackers.tracker_api import Tracker
from submodules.AlphaPose.trackers.tracker_cfg import cfg as tcfg
from submodules.AlphaPose.trackers import track
from submodules.AlphaPose.alphapose.models import builder
from submodules.AlphaPose.alphapose.utils.config import update_config
from submodules.AlphaPose.alphapose.utils.detector import DetectionLoader
from submodules.AlphaPose.alphapose.utils.file_detector import FileDetectionLoader  # noqa
from submodules.AlphaPose.alphapose.utils.transforms import flip, flip_heatmap
from submodules.AlphaPose.alphapose.utils.vis import getTime
from submodules.AlphaPose.alphapose.utils.webcam_detector import WebCamDetectionLoader  # noqa
from submodules.AlphaPose.alphapose.utils.writer import DataWriter

from alphapose_.skeleton import AlphaPosePoseExtractor
from alphapose_.utils import check_input
from alphapose_.utils import loop
from alphapose_.utils import print_finish_info


def parse_args():
    from alphapose_.args import args
    if platform.system() == 'Windows':
        args.sp = True
    if torch.cuda.device_count() >= 1:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        args.device = torch.device("cuda:" + str(args.gpus[0]))
    else:
        args.gpus = [-1]
        args.device = torch.device("cpu")
    args.detbatch = args.detbatch * len(args.gpus)
    args.posebatch = args.posebatch * len(args.gpus)
    args.tracking = \
        args.pose_track or args.pose_flow or args.detector == 'tracker'
    return args


if __name__ == "__main__":

    # 1. Checks + parse the args
    args = parse_args()

    if args.outputpath != '-1':
        os.makedirs(args.outputpath, exist_ok=True)

    # 2.Checks the in and outputs
    # 3. Load detection loader
    # 4. Load pose model
    # 5. Init data writer
    # 6. Load tracker
    PE = AlphaPosePoseExtractor(args)

    # 7. Init the loop
    if PE.input_mode == 'webcam':
        print('Starting webcam demo, press Ctrl + C to terminate...')
        sys.stdout.flush()
        im_names_desc = tqdm(loop())
    else:
        data_len = PE.det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    # 8. Main loop
    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)
    try:
        for i in im_names_desc:
            with torch.no_grad():
                PE.predict()
            if args.profile:
                # TQDM
                im_names_desc.set_description(
                    f"det time: {np.mean(PE.runtime_profile['dt']):.4f} | "
                    f"pose time: {np.mean(PE.runtime_profile['pt']):.4f} | "
                    f"post processing: {np.mean(PE.runtime_profile['pn']):.4f}"
                )
        print_finish_info(args)
        while (PE.writer.running()):
            time.sleep(1)
            print('===========================> Rendering remaining ' +
                  str(PE.writer.count()) + ' images in the queue...')
        PE.writer.stop()
        PE.det_loader.stop()
    except Exception as e:
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')   # noqa
        pass
    except KeyboardInterrupt:
        print_finish_info(args)
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            PE.det_loader.terminate()
            while (PE.writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' +
                      str(PE.writer.count()) + ' images in the queue...')
            PE.writer.stop()
        else:
            # subprocesses are killed, manually clear queues
            PE.det_loader.terminate()
            PE.writer.terminate()
            PE.writer.clear_queues()
            PE.det_loader.clear_queues()

    print("FINISH...")
