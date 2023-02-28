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

    # 1. Checks + parse the args + config
    args = parse_args()
    cfg = update_config(args.cfg)

    # 2.Checks the in and outputs
    mode, input_source = check_input(args)
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # 3. Load detection loader
    if mode == 'webcam':
        det_loader = WebCamDetectionLoader(
            input_source, get_detector(args), cfg, args)
        det_worker = det_loader.start()
    elif mode == 'detfile':
        det_loader = FileDetectionLoader(input_source, cfg, args)
        det_worker = det_loader.start()
    else:
        det_loader = DetectionLoader(
            input_source, get_detector(args), cfg, args,
            batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
        det_worker = det_loader.start()

    # 4. Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(
        args.checkpoint, map_location=args.device))
    dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(
            pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()

    # 5. Load tracker
    if args.pose_track:
        tracker = Tracker(tcfg, args)

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # 6. Init data writer
    queueSize = 2 if mode == 'webcam' else args.qsize
    if args.save_video and mode != 'image':
        from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt  # noqa
        if mode == 'video':
            video_save_opt['savepath'] = os.path.join(
                args.outputpath, 'AlphaPose_' + os.path.basename(input_source))
        else:
            video_save_opt['savepath'] = os.path.join(
                args.outputpath, 'AlphaPose_webcam' + str(input_source) + '.mp4')  # noqa
        video_save_opt.update(det_loader.videoinfo)
        writer = DataWriter(cfg, args, save_video=True,
                            video_save_opt=video_save_opt,
                            queueSize=queueSize).start()
    else:
        writer = DataWriter(cfg, args, save_video=False,
                            queueSize=queueSize).start()

    # 7. Init the loop
    if mode == 'webcam':
        print('Starting webcam demo, press Ctrl + C to terminate...')
        sys.stdout.flush()
        im_names_desc = tqdm(loop())
    else:
        data_len = det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    # 8. Main loop
    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)
    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores,
                 ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None,
                                None, orig_img, im_name)
                    continue
                if args.profile:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]  # noqa
                    if args.flip:
                        inps_j = torch.cat((inps_j, flip(inps_j)))
                    hm_j = pose_model(inps_j)
                    if args.flip:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], dataset.joint_pairs, shift=True)  # noqa
                        hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)
                hm = torch.cat(hm)
                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)
                if args.pose_track:
                    boxes, scores, ids, hm, cropped_boxes = track(
                        tracker, args, orig_img, inps,
                        boxes, hm, cropped_boxes, im_name, scores)
                hm = hm.cpu()
                writer.save(boxes, scores, ids, hm,
                            cropped_boxes, orig_img, im_name)
                if args.profile:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)

            if args.profile:
                # TQDM
                im_names_desc.set_description(
                    f"det time: {np.mean(runtime_profile['dt']):.4f} | "
                    f"pose time: {np.mean(runtime_profile['pt']):.4f} | "
                    f"post processing: {np.mean(runtime_profile['pn']):.4f}"
                )
        print_finish_info(args)
        while (writer.running()):
            time.sleep(1)
            print('===========================> Rendering remaining ' +
                  str(writer.count()) + ' images in the queue...')
        writer.stop()
        det_loader.stop()
    except Exception as e:
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')   # noqa
        pass
    except KeyboardInterrupt:
        print_finish_info(args)
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            det_loader.terminate()
            while (writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' +
                      str(writer.count()) + ' images in the queue...', end='\r')
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues

            det_loader.terminate()
            writer.terminate()
            writer.clear_queues()
            det_loader.clear_queues()
