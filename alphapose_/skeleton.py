import argparse
import os
import torch
from torch.nn import DataParallel

from submodules.AlphaPose.detector.apis import get_detector
from submodules.AlphaPose.trackers.tracker_api import Tracker
from submodules.AlphaPose.trackers.tracker_cfg import cfg as tcfg
from submodules.AlphaPose.trackers import track
from submodules.AlphaPose.alphapose.models.builder import build_sppe
from submodules.AlphaPose.alphapose.models.builder import retrieve_dataset
from submodules.AlphaPose.alphapose.utils.config import update_config
from submodules.AlphaPose.alphapose.utils.detector import DetectionLoader
from submodules.AlphaPose.alphapose.utils.file_detector import FileDetectionLoader  # noqa
from submodules.AlphaPose.alphapose.utils.transforms import flip, flip_heatmap
from submodules.AlphaPose.alphapose.utils.vis import getTime
from submodules.AlphaPose.alphapose.utils.webcam_detector import WebCamDetectionLoader  # noqa
from submodules.AlphaPose.alphapose.utils.writer import DataWriter
from submodules.AlphaPose.alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT

from alphapose_.utils import check_input
from alphapose_.utils import loop
from alphapose_.utils import print_finish_info


class AlphaPosePoseExtractor:

    def __init__(self, args: argparse.Namespace) -> None:
        cfg = update_config(args.cfg)
        print(f"Building pose model: {cfg.MODEL.TYPE}")
        print(f"Loading pose model checkpoint: {args.checkpoint}")
        self.args = args
        self.cfg = cfg
        self.dataset = retrieve_dataset(cfg.DATASET.TRAIN)
        self.batchSize = self.args.posebatch
        if self.args.flip:
            self.batchSize = int(self.batchSize / 2)
        self.pose_model = None  # nn.Module
        self.det_loader = None  # Loader class
        self.det_worker = None  # list
        self.input_mode = None  # str
        self.input_source = None  # str
        self.writer = None
        if self.args.pose_track:
            print(f"Building pose track model")
            print(f"Loading pose track checkpoint: {tcfg.loadmodel}")
            tcfg.loadmodel = args.pose_track_model
            self.tracker = Tracker(tcfg, args)
        self._build_pose_model()
        self._build_detection_loader()
        self._build_writer()
        self.runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }

    def _build_pose_model(self):
        self.pose_model = build_sppe(self.cfg.MODEL,
                                     preset_cfg=self.cfg.DATA_PRESET)
        self.pose_model.load_state_dict(
            torch.load(self.args.checkpoint, map_location=self.args.device))
        if len(self.args.gpus) > 1:
            self.pose_model = DataParallel(self.pose_model,
                                           device_ids=self.args.gpus)
        self.pose_model.to(self.args.device)
        self.pose_model.eval()

    def _build_detection_loader(self):
        self.input_mode, self.input_source = check_input(self.args)
        if self.input_mode == 'webcam':
            self.det_loader = WebCamDetectionLoader(
                input_source=self.input_source,
                cfg=self.cfg,
                opt=self.args,
                detector=get_detector(self.args)
            )
        elif self.input_mode == 'detfile':
            self.det_loader = FileDetectionLoader(
                input_source=self.input_source,
                cfg=self.cfg,
                opt=self.args
            )
        else:
            self.det_loader = DetectionLoader(
                input_source=self.input_source,
                cfg=self.cfg,
                opt=self.args,
                detector=get_detector(self.args),
                batchSize=self.args.detbatch,
                mode=self.input_mode,
                queueSize=self.args.qsize,
            )
        self.det_worker = self.det_loader.start()

    def _build_writer(self):
        queueSize = 2 if self.input_mode == 'webcam' else self.args.qsize
        opt = DEFAULT_VIDEO_SAVE_OPT
        if self.args.save_video and self.input_mode != 'image':
            opt['savepath'] = os.path.join(self.args.outputpath, 'AlphaPose_')
            if self.input_mode == 'video':
                opt['savepath'] += os.path.basename(self.input_source)
            else:
                opt['savepath'] += 'webcam' + str(self.input_source) + '.mp4'
            opt.update(self.det_loader.videoinfo)
            self.writer = DataWriter(self.cfg, self.args, save_video=True,
                                     video_save_opt=opt,
                                     queueSize=queueSize).start()
        else:
            self.writer = DataWriter(self.cfg, self.args, save_video=False,
                                     video_save_opt=opt,
                                     queueSize=queueSize).start()

    def detect(self) -> tuple:
        return self.det_loader.read()

    def save(self, *args, **kwargs):
        if self.args.outputpath != '-1':
            self.writer.save(*args, **kwargs)

    def predict(self):
        # return 0,1 = break,continue

        if self.args.profile:
            start_time = getTime()

        (inps, orig_img, im_name,
            boxes, scores, ids, cropped_boxes) = self.detect()
        if orig_img is None:  # NO INPUT DATA
            return 0
        if boxes is None or boxes.nelement() == 0:
            self.save(None, None, None, None, None, orig_img, im_name)
            return 1

        if self.args.profile:
            ckpt_time, det_time = getTime(start_time)
            self.runtime_profile['dt'].append(det_time)

        # Pose Estimation
        inps = inps.to(self.args.device)
        datalen = inps.size(0)
        leftover = 0
        if (datalen) % self.batchSize:
            leftover = 1
        num_batches = datalen // self.batchSize + leftover
        hm = []
        for j in range(num_batches):
            inps_j = inps[j * self.batchSize:min((j + 1) * self.batchSize, datalen)]  # noqa
            if self.args.flip:
                inps_j = torch.cat((inps_j, flip(inps_j)))
            hm_j = self.pose_model(inps_j)
            if self.args.flip:
                hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):],
                                         self.dataset.joint_pairs,
                                         shift=True)
                hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
            hm.append(hm_j)
        hm = torch.cat(hm)

        if self.args.profile:
            ckpt_time, pose_time = getTime(ckpt_time)
            self.runtime_profile['pt'].append(pose_time)

        # Pose Track
        if self.args.pose_track:
            boxes, scores, ids, hm, cropped_boxes = track(
                self.tracker, self.args, orig_img, inps,
                boxes, hm, cropped_boxes, im_name, scores)

        hm = hm.cpu()
        self.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)

        if self.args.profile:
            ckpt_time, post_time = getTime(ckpt_time)
            self.runtime_profile['pn'].append(post_time)

        return 1
