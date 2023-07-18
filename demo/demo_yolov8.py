import cv2
import numpy as np
import time
import torch

# ultralytics.yolo.cfg.default.yaml
from ultralytics import YOLO

from typing import Callable

from rs_py.rs_run_devices import get_rs_parser
from rs_py.rs_run_devices import printout

from openpose.native.python.skeleton import PyOpenPoseNative
from openpose.native.python.utils import Timer

from demo import App as AppBase
from demo import TextParser, ActionHistory, Storage, transform_3dpose


FPS = 30
CAMERA_W = 848
CAMERA_H = 480

# HORITONTAL_BAR = np.ones((30, CAMERA_W, 3), dtype=np.uint8)*220
VERTICAL_BAR = np.concatenate(
    [np.ones((CAMERA_W, 5, 3), dtype=np.uint8)*np.array((255, 150, 0), dtype=np.uint8),  # noqa
     np.ones((CAMERA_W, 5, 3), dtype=np.uint8)*np.array((0, 0, 0), dtype=np.uint8),  # noqa
    #  np.ones((CAMERA_W, 6, 3), dtype=np.uint8)*np.array((255, 150, 0), dtype=np.uint8),  # noqa
    #  np.ones((CAMERA_W, 6, 3), dtype=np.uint8)*np.array((255, 255, 255), dtype=np.uint8),  # noqa
     np.ones((CAMERA_W, 5, 3), dtype=np.uint8)*np.array((255, 150, 0), dtype=np.uint8)],  # noqa
    axis=1
)

FIG, POSE, EDGE = None, None, None
MATPLOT = False

# # # cannot import this with opencv
# # # https://github.com/fourMs/MGT-python/issues/200
# import matplotlib.pyplot as plt
# FIG = plt.figure(figsize=(16, 8))
# MATPLOT = True

# OUTPUT_VIDEO = cv2.VideoWriter(
#     'project_yolov8.avi', cv2.VideoWriter_fourcc(*'DIVX'),
#     30,
#     (1920, 1080)
# )
OUTPUT_VIDEO = None


def get_args_rs():
    args, _ = get_rs_parser().parse_known_args()
    args.rs_steps = 1
    args.rs_fps = FPS
    args.rs_image_width = CAMERA_W
    args.rs_image_height = CAMERA_H
    args.rs_ir_emitter_power = 290
    args.rs_display_frame = 1
    args.rs_save_with_key = False
    args.rs_save_data = False
    args.rs_save_path = ''
    args.rs_postprocess = True  # max fps ~25 for 1280x720
    args.rs_vertical = True
    print("========================================")
    print(">>>>> args_rs <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")
    return args


class Config():
    def __init__(self) -> None:
        # args
        self.args_rs = None
        self.args_op = None
        self.args_ar = None
        # Delay = predict and no update in tracker
        self.delay_switch = 2
        self.delay_counter = 2
        # For cv.imshow
        self.display_speed = 1
        # Runtime logging
        self.enable_timer = True
        self.fps = {'PE': [], 'TK': []}
        self.infer_time = -1
        self.track_time = -1
        self.filered_skel = -1
        self.prep_time = -1
        self.recog_time = -1
        self.rs_time = -1
        # Storage --------------------------------------------------------------
        # max length of data to store
        self.MAX_STORAGE_LEN = 100
        # buffer before removed
        self.BUFFER = 0
        # moving average and threshold for average
        self.MOVING_AVG = 9
        self.SCORE_THRES = 0.25
        # Motion and action filtering
        self.MOTION_DELAY = 5
        self.MOTION_COUNT = 10
        self.ACTION_COUNT = 5
        self.MAY_YS_DELAY = 6
        self.MAY_YS_COUNT = 9
        self.MAY_YL_DELAY = 18
        self.MAY_YL_COUNT = 21
        # TextParser -----------------------------------------------------------
        self.MOVE_THRES = 0.015
        # ActionHistory --------------------------------------------------------
        self.LOG_LEFT_OFFSET = 20
        self.LOG_TOP_OFFSET = 20
        self.LOG_HEIGHT = 14
        self.LOG_WIDTH = 30
        self.LOG_NUM_LB = 700//self.LOG_HEIGHT
        self.LOG_LB_OFFSET = 20
        self.BAR_LEFT_OFFSET = 0
        self.BAR_TOP_OFFSET = CAMERA_W - 40
        self.BAR_HEIGHT = 20
        self.BAR_WIDTH = 1
        self.BAR_NUM_LB = CAMERA_H*3//self.BAR_WIDTH
        self.BAR_LB_OFFSET = 20
        # Rule based action ----------------------------------------------------
        self.RULE_THRES_STAND = 0.05
        self.RULE_THRES_STAND_LONG = 0.10
        self.RULE_THRES_SIT = -0.05
        self.RULE_THRES_SIT_LONG = -0.09
        self.RULE_THRES_FALL = -0.16
        self.RULE_THRES_FALL_LONG = -0.25
        self.RULE_THRES_FALL_HEIGHT = 1.19


class App(AppBase):
    def __init__(self) -> None:
        super().__init__()
        self.CF = None
        self.DS = None
        self.TP = None
        self.AH = None
        self.ET = None
        self.AR = None
        self.LABEL_MAPPING = None
        self.RS = None
        self.RS_INFO = {}

        self.valid_op_image = [0, None]

        self.YOLO_SEG = YOLO("./yolov8n-seg.pt")
        self.YOLO_POS = YOLO("./yolov8n-pose.pt")

    def setup(self,
              config: Config,
              get_args_rs_func: Callable = get_args_rs,
              ) -> None:
        # Overall configurations
        self.CF = config
        self.CF.args_rs = get_args_rs_func()
        # For skeleton storage
        self.DS = Storage(self.CF)
        # For parsing sentences
        self.TP = TextParser(self.CF, fps=self.CF.args_rs.rs_fps)
        # Tracks previous action predictionsd
        self.AH = ActionHistory(self.CF)

    def infer_pose_and_track(self, color_image, depth_image):
        if not hasattr(self.YOLO_POS.predictor, 'trackers'):
            from ultralytics.tracker import register_tracker
            register_tracker(self.YOLO_POS, persist=True)

        if self.CF.args_rs.rs_vertical:
            rot_color_image = np.rot90(color_image, 1).copy()
        else:
            rot_color_image = color_image
        intr_mat = self.RS_INFO['intr_mat']

        # To change color: ultralytics.yolo.utils.plotting.py
        results_seg = self.YOLO_SEG(
            source=rot_color_image.copy(),
            imgsz=640,
            classes=[0, 39, 41, 45, 73],
            max_det=10,
            iou=0.5,
            conf=0.25,
            retina_masks=True,
            show=False,
            device=0,
            verbose=False
        )
        results_pos = self.YOLO_POS(
            source=rot_color_image.copy(),
            imgsz=640,
            classes=0,
            max_det=10,
            iou=0.5,
            conf=0.25,
            mode='track',
            # persist=True,
            show=False,
            device=0,
            verbose=False
        )
        pose_image = np.zeros_like(rot_color_image)
        skel_added = False
        result_seg = results_seg[0].cpu()
        result_pos = results_pos[0].cpu()
        keep_obj = []
        for i in range(result_pos.keypoints.data.shape[0]):
            if result_pos.boxes.data.numel() == 0:
                continue
            if result_pos.boxes.id is None:
                continue
            pose_result = result_pos.keypoints.data[i].numpy()
            pose_result = np.stack(
                [
                    CAMERA_W - pose_result[:, 1],
                    # pose_result[:, 1],
                    pose_result[:, 0],
                    pose_result[:, 2]
                ],
                -1
            )
            joints3d = PyOpenPoseNative.get_3d_skeleton(
                skeleton=pose_result,
                depth_img=depth_image,
                intr_mat=intr_mat,
                depth_scale=self.RS_INFO['depth_scale'],
                patch_offset=3
            )
            joints3d = transform_3dpose(joints3d,
                                        self.RS_INFO['depth_scale'])
            if joints3d[[5, 6, 11, 12], 2].max() > 5.2:
                continue
            # print(joints3d[:, 2].max())
            # print(pose_result)
            self.DS.add(
                result_pos.boxes.id[i].item(),
                pose_result[:, :2],
                joints3d,
                pose_result[:, 2]
            )
            skel_added = True
            keep_obj.append(i)

        if ((result_seg.masks is not None) and
            (result_seg.boxes.data.numel() > 0) and
            (result_pos.boxes.data.numel() > 0) and
            (result_pos.keypoints is not None)):  # noqa
            try:
                result_pos.boxes.data = result_pos.boxes.data[keep_obj, :]
            except:
                None
            try:
                result_pos.keypoints.data = result_pos.keypoints.data[keep_obj, :]
            except:
                None
            # try:
            #     result_seg.boxes.data = result_seg.boxes.data[keep_obj, :]
            # except:
            #     None
            # try:
            #     result_seg.masks.data = result_seg.masks.data[keep_obj, :]
            # except:
            #     None

        pose_image = result_seg.plot(img=pose_image,
                                     boxes=False,
                                     masks=True)
        pose_image = result_pos.plot(img=pose_image,
                                     boxes=True,
                                     masks=False)

        # print([round(i, 2) for i in result_pos.keypoints.conf.tolist()[0]])
        self.DS.check_valid_and_delete()

        if self.CF.args_rs.rs_vertical:
            pose_image = np.rot90(pose_image, -1).copy()

        return True, pose_image, skel_added


if __name__ == "__main__":

    config = Config()
    app = App()
    app.setup(config)

    cv2.namedWindow("Monitor", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Monitor",
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Setup realsense ----------------------------------------------------------
    app.setup_realsense()

    # MAIN LOOP ----------------------------------------------------------------
    ar_start_time = time.time()

    # try:
    c = 0
    max_c = int(1e8)

    while True:

        if c < 15:
            fps = {'PE': [], 'TK': []}

        # 1. grab image from realsense -------------------------------------
        with Timer("rsw", app.CF.enable_timer, False) as t:
            app.RS.step(
                display=0,
                display_and_save_with_key=False,
                use_colorizer=False
            )
        app.CF.rs_time = t.duration
        frame = app.RS.frames[app.RS_INFO['device_sn']]
        ori_color_image = frame['color_framedata'].copy()  # h,w,c
        color_image = frame['color_framedata']  # h,w,c
        depth_image = frame['depth_framedata']  # h,w
        if app.CF.args_rs.rs_postprocess:
            depth_image = cv2.resize(depth_image, (color_image.shape[1],
                                                   color_image.shape[0]))
        # Remove the border pixels as measurements there are not good
        color_image[:20, :, :] = 169  # right border
        color_image[-20:, :, :] = 169  # left border
        color_image[depth_image > 4000] = 169
        color_image[depth_image < 300] = 169
        # depth_colormap = rsw.frames[device_sn]['depth_color_framedata']
        # quit_key = app.visualize_rs(color_image=color_image,
        #                             depth_colormap=depth_colormap,
        #                             depth_scale=depth_scale,
        #                             winname=f'rs_{device_sn}')
        # if quit_key:
        #     break

        # 2. track without pose extraction ---------------------------------
        # 3. infer pose and track ------------------------------------------
        status, op_image, skel_added = app.infer_pose_and_track(
            color_image, depth_image)

        if not status:
            break

        # 4. action recognition --------------------------------------------
        # if time.time() - ar_start_time < int(app.CF.args_ar.interval):
        #     continue

        distance_str = ""
        raw_kypts = None
        prediction = -1
        motion = 0
        with Timer("AR", app.CF.enable_timer, False) as t:
            # if len(app.DS.skeletons) > 0:
            if skel_added:
                (raw_kypts, raw_score, avg_skels,
                    motion, prediction, distance_str) = app.infer_action(
                        c,
                        [5, 6, 11, 12]
                )
            else:
                printout(f'{c:04} No skeletons... ', 'i')

        app.CF.recog_time = t.duration
        if app.CF.recog_time is None:
            app.CF.recog_time = -1

        # Visualize results -----------
        if not MATPLOT:
            depth_image[depth_image > 3500] = 3500
            depth_image[depth_image < 300] = 300
            image = app.visualize_output(
                ori_color_image,
                depth_image,
                op_image,
                raw_kypts,
                prediction,
                motion,
                distance_str,
                app.CF.args_rs.rs_vertical
            )
            # Face sensor
            try:
                raw_kypts = app.DS.get_last_skel(None)[0]
            except ValueError:
                raw_kypts = None
            if raw_kypts is not None:
                pos = raw_kypts.astype(int)[:, 0, :5, :]  # [M, V, 2]
                # pos = raw_kypts.astype(int)[:, 0, [5, 6, 11, 12], :]
                if app.CF.args_rs.rs_vertical:
                    pos = np.flip(pos)
                    pos[:, :, 1] = CAMERA_W - pos[:, :, 1]
                for pos_i in pos:
                    for pos_i_v in pos_i:
                        cv2.circle(image, pos_i_v, 20, (255, 255, 255), -1)

            # Unblur blur due to resize to fullscreen:
            # https://stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencv
            image = cv2.resize(image, (1920, 1080))
            blurr = cv2.GaussianBlur(image, (0, 0), 3)
            image = cv2.addWeighted(image, 1.5, blurr, -0.5, 0)

            # print(image.shape)
            if OUTPUT_VIDEO is not None:
                OUTPUT_VIDEO.write(image)

            cv2.imshow("Monitor", image)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                break

        if MATPLOT:
            # M, 1, V, C -> C, 1, V, M
            avg_skels = avg_skels.transpose(3, 1, 2, 0)[:, :, :, 0:1]  # noqa
            avg_skels[1] *= -1.0
            alpha = raw_score[0, 0].tolist()
            num_joint = app.CF.args_ar.num_joint
            app.visualize_3dplot(avg_skels, alpha, num_joint)

        if not len(app.RS.frames) > 0:
            printout(f"Empty...", 'w')
            continue

        if c % (app.CF.args_rs.rs_fps*60) == 0 and c > 0:
            app.DS.reset()

        c += 1

    # except Exception as e:
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     printout(f"{exc_type}, {fname}, {exc_tb.tb_lineno}", 'e')
    #     printout(f"Exception msg : {e}", 'e')
    #     traceback.print_tb(exc_tb)
    #     printout(f"Stopping RealSense devices...", 'i')
    #     app.RS.stop()

    # finally:
    #     printout(f"Final RealSense devices...", 'i')
    #     app.RS.stop()

    app.RS.stop()

    if OUTPUT_VIDEO is not None:
        OUTPUT_VIDEO.release()
    printout(f"Finished...", 'i')
