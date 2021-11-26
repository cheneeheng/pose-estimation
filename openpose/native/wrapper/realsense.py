import cv2
import json
import numpy as np
import os
import pyrealsense2 as rs


def read_realsense_calibration(file_path):

    class RealsenseConfig:
        def __init__(self, json_file):
            self.width = json_file['rgb'][0]['width']
            self.height = json_file['rgb'][0]['height']
            self.rgb_intrinsics = np.array(json_file['rgb'][0]['intrinsic_mat']).reshape(3, 3)  # noqa
            self.depth_intrinsics = np.array(json_file['depth'][0]['intrinsic_mat']).reshape(3, 3)  # noqa
            self.depth_scale = json_file['depth'][0]['depth_scale']
            self.T_rgb_depth = np.eye(4)
            self.T_rgb_depth[:3, :3] = np.array(json_file['T_rgb_depth'][0]['rotation']).reshape(3, 3)  # noqa
            self.T_rgb_depth[:3, 3] = json_file['T_rgb_depth'][0]['translation']  # noqa

    with open(file_path) as calib_file:
        calib = json.load(calib_file)
    return RealsenseConfig(calib)


class RealsenseWrapper(object):

    def __init__(self) -> None:
        super().__init__()
        self.cfg = None
        self.pipeline = None
        self.profile = None
        self.calib_data = None

    def save_calibration(self, save_path: str) -> None:
        """Assumes that realsense is mounted statically. """

        # Intrinsics of RGB & depth frames
        profile_rgb = self.profile.get_stream(rs.stream.color)
        intr_rgb = profile_rgb.as_video_stream_profile().get_intrinsics()
        # intr_rgb_mat = np.array([[intr_rgb.fx, 0, intr_rgb.ppx],
        #                         [0, intr_rgb.fy, intr_rgb.ppy],
        #                         [0, 0, 1]])

        # Fetch stream profile for depth stream
        profile_depth = self.profile.get_stream(rs.stream.depth)

        # Downcast to video_stream_profile and fetch intrinsics
        intr_depth = profile_depth.as_video_stream_profile().get_intrinsics()
        # intr_depth_mat = np.array([[intr_depth.fx, 0, intr_depth.ppx],
        #                            [0, intr_depth.fy, intr_depth.ppy],
        #                            [0, 0, 1]])

        # Extrinsic matrix from RGB sensor to Depth sensor
        extr = profile_rgb.as_video_stream_profile().get_extrinsics_to(profile_depth)  # noqa
        extr_mat = np.eye(4)
        extr_mat[:3, :3] = np.array(extr.rotation).reshape(3, 3)
        extr_mat[:3, 3] = extr.translation

        # Depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        # print("Depth Scale is: ", depth_scale)

        # Write calibration data to json file
        calib_data = {}
        calib_data['rgb'] = []
        calib_data['rgb'].append({
            'width': intr_rgb.width,
            'height': intr_rgb.height,
            'intrinsic_mat': [intr_rgb.fx, 0, intr_rgb.ppx,
                              0, intr_rgb.fy, intr_rgb.ppy,
                              0, 0, 1]
        })
        calib_data['depth'] = []
        calib_data['depth'].append({
            'width': intr_depth.width,
            'height': intr_depth.height,
            'intrinsic_mat': [intr_depth.fx, 0, intr_depth.ppx,
                              0, intr_depth.fy, intr_depth.ppy,
                              0, 0, 1],
            'depth_scale': depth_scale
        })
        calib_data['T_rgb_depth'] = []
        calib_data['T_rgb_depth'].append({
            'rotation': extr.rotation,
            'translation': extr.translation
        })

        self.calib_data = calib_data

        assert os.path.exists(save_path)
        if os.path.isfile(save_path):
            with open(save_path, 'w') as outfile:
                json.dump(calib_data, outfile, indent=4)
        else:
            with open(os.path.join(save_path, 'calib.txt'), 'w') as outfile:
                json.dump(calib_data, outfile, indent=4)

    def configure(self) -> None:
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        self.cfg = cfg

    def initialize(self) -> None:
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(self.cfg)
        # align depth to rgb frame
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def run(self,
            rgb_save_path: str = None,
            depth_save_path: str = None,
            timestamp_file=None,
            display: bool = False) -> bool:

        # while True:
        frames = self.pipeline.wait_for_frames()
        timestamp = frames.get_frame_metadata(
            rs.frame_metadata_value.sensor_timestamp)

        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not aligned_color_frame:
            return False, None, None

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())

        if depth_save_path is not None:
            np.save(os.path.join(depth_save_path, f'{timestamp}'), depth_image)

        if rgb_save_path is not None:
            np.save(os.path.join(rgb_save_path, f'{timestamp}'), color_image)

        if timestamp_file is not None:
            with open(timestamp_file, 'a+') as f:
                f.write(f'{timestamp}\n')

        if display:
            # Render images
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET)

            # # Set pixels further than clipping_distance to grey
            # clipping_distance = 10
            # grey_color = 153
            # # depth image is 1 channel, color is 3 channels
            # depth_image_3d = np.dstack(
            #     (depth_image, depth_image, depth_image))
            # bg_removed = np.where(
            #     (depth_image_3d > clipping_distance) | (
            #         depth_image_3d <= 0), grey_color, color_image)

            # images = np.hstack((bg_removed, depth_colormap))
            # images = np.hstack((color_image, depth_colormap))

            images_overlapped = cv2.addWeighted(
                color_image, 0.3, depth_colormap, 0.5, 0)
            images = np.hstack(
                (color_image, depth_colormap, images_overlapped))

            cv2.namedWindow('Align overlapped', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align overlapped', images)
            key = cv2.waitKey(30)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                cv2.waitKey(5)
                return False, None, None

        return True, color_image, depth_image, timestamp
