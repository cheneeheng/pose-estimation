import cv2
import json
import numpy as np
import os
import pyrealsense2 as rs

from typing import Optional

from realsense_device_manager import Device
from realsense_device_manager import enumerate_connected_devices
from realsense_device_manager import post_process_depth_frame


def read_realsense_calibration(file_path: str):

    class RealsenseConfig:
        def __init__(self, json_data: dict):
            self.width = json_data['color'][0]['width']
            self.height = json_data['color'][0]['height']
            self.color_intrinsics = np.array(json_data['color'][0]['intrinsic_mat']).reshape(3, 3)  # noqa
            self.depth_intrinsics = np.array(json_data['depth'][0]['intrinsic_mat']).reshape(3, 3)  # noqa
            self.depth_scale = json_data['depth'][0]['depth_scale']
            self.T_color_depth = np.eye(4)
            self.T_color_depth[:3, :3] = np.array(json_data['T_color_depth'][0]['rotation']).reshape(3, 3)  # noqa
            self.T_color_depth[:3, 3] = json_data['T_color_depth'][0]['translation']  # noqa

    with open(file_path) as calib_file:
        calib = json.load(calib_file)
    return RealsenseConfig(calib)


class StoragePaths(object):
    def __init__(self):
        self.calib = None
        self.color = None
        self.depth = None
        self.skeleton = None
        self.timestamp = None
        self.timestamp_file = None


class RealsenseWrapper(object):

    def __init__(self) -> None:
        super().__init__()
        self._cfg = {}
        self.enabled_devices = {}  # serial numbers of enabled devices
        self._align = None
        self.calib_data = {}
        self.fps = 30
        self.height = 480
        self.width = 848

    def configure(self,
                  device_sn: Optional[str] = None,
                  fps: Optional[int] = None,
                  height: Optional[int] = None,
                  width: Optional[int] = None) -> None:
        """Defines per device configurations.

        device('001622070408')
        device('001622070717')

        Args:
            device_sn (str, optional): serial number. Defaults to None.
            fps (int, optional): frame fps. Defaults to 30.
            height (int, optional): frame height. Defaults to 480.
            width (int, optional): frame width. Defaults to 848.
        """
        if device_sn is not None:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth,
                              width if width is not None else self.width,
                              height if height is not None else self.height,
                              rs.format.z16,
                              fps if fps is not None else self.fps)
            cfg.enable_stream(rs.stream.color,
                              width if width is not None else self.width,
                              height if height is not None else self.height,
                              rs.format.bgr8,
                              fps if fps is not None else self.fps)
            self._cfg[device_sn] = cfg

    def initialize(self, enable_ir_emitter: bool = True) -> None:

        if len(self._cfg) == 0:
            self.configure('default')

        available_devices = enumerate_connected_devices(rs.context())
        for device_serial, product_line in available_devices:
            pipeline = rs.pipeline()
            cfg = self._cfg.get(device_serial, self._cfg['default'])
            cfg.enable_device(device_serial)
            pipeline_profile = pipeline.start(cfg)

            depth_sensor = pipeline_profile.get_device().first_depth_sensor()
            if enable_ir_emitter:
                if depth_sensor.supports(rs.option.emitter_enabled):
                    depth_sensor.set_option(rs.option.emitter_enabled,
                                            1 if enable_ir_emitter else 0)
                    # depth_sensor.set_option(rs.option.laser_power, 330)

            self.enabled_devices[device_serial] = (
                Device(pipeline, pipeline_profile, product_line))

        # align depth to color frame
        self._align = rs.align(rs.stream.color)

    def run(self,
            storage_paths: Optional[StoragePaths] = None,
            display: bool = False) -> bool:

        if len(self.enabled_devices) == 0:
            print("No devices are enabled...")
            return {}

        frames = {}
        while len(frames) < len(self.enabled_devices.items()):

            for dev_sn, dev in self.enabled_devices.items():

                streams = dev.pipeline_profile.get_streams()
                # frameset will be a pyrealsense2.composite_frame object
                frameset = dev.pipeline.poll_for_frames()

                if frameset.size() == len(streams):
                    frames[dev_sn] = {}

                    frames[dev_sn]['calib'] = self.calib_data[dev_sn]

                    timestamp = frameset.get_frame_metadata(
                        rs.frame_metadata_value.sensor_timestamp)
                    frames[dev_sn]['timestamp'] = timestamp

                    ts_file = storage_paths[dev_sn].timestamp_file
                    if ts_file is not None:
                        with open(ts_file, 'a+') as f:
                            f.write(f'{timestamp}\n')

                    aligned_frameset = self._align.process(frameset)
                    for stream in streams:
                        st = stream.stream_type()
                        # if stream.stream_type() == rs.stream.infrared:
                        #     frame = aligned_frameset.get_infrared_frame(
                        #         stream.stream_index())
                        #     key_ = (stream.stream_type(),
                        #             stream.stream_index())
                        # frame = aligned_frameset.first_or_default(st)
                        # frame_data = frame.get_data()
                        # frames[dev_sn][st] = frame_data
                        if st == rs.stream.color:
                            frame = aligned_frameset.first_or_default(st)
                            frame_data = frame.get_data()
                            frames[dev_sn]['color'] = frame_data
                            filepath = storage_paths[dev_sn].color
                            if filepath is not None:
                                np.save(os.path.join(filepath, f'{timestamp}'),
                                        frame_data)
                        elif st == rs.stream.depth:
                            frame = aligned_frameset.first_or_default(st)
                            frame = post_process_depth_frame(frame)
                            frame_data = frame.get_data()
                            frames[dev_sn]['depth'] = frame_data
                            filepath = storage_paths[dev_sn].depth
                            if filepath is not None:
                                np.save(os.path.join(filepath, f'{timestamp}'),
                                        frame_data)

            if display:

                for dev_sn, data_dict in frames.items():

                    # Render images
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(data_dict['depth'], alpha=0.03),
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
                        data_dict['color'], 0.3, depth_colormap, 0.5, 0)
                    images = np.hstack(
                        (data_dict['color'], depth_colormap, images_overlapped))

                    cv2.namedWindow(f'{dev_sn}', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow(f'{dev_sn}', images)
                    key = cv2.waitKey(30)
                    # Press esc or 'q' to close the image window
                    if key & 0xFF == ord('q') or key == 27:
                        cv2.destroyAllWindows()
                        cv2.waitKey(5)
                        return {}

            return frames

    def stop(self) -> None:
        """Stops the devices. """
        if len(self.enabled_devices) == 0:
            print("No devices are enabled...")
        else:
            for _, dev in self.enabled_devices.items():
                dev.pipeline.stop()

    def save_calibration(self, storage_paths: StoragePaths) -> None:
        """Saves camera calibration.

        Args:
            save_path (dict): path to save the calibration data.
        """

        if len(self.enabled_devices) == 0:
            print("No devices are enabled...")
            return

        for dev_sn, dev in self.enabled_devices.items():
            profile = dev.pipeline_profile

            # Intrinsics of color & depth frames -------------------------------
            profile_color = profile.get_stream(rs.stream.color)
            intr_color = profile_color.as_video_stream_profile()
            intr_color = intr_color.get_intrinsics()

            # Fetch stream profile for depth stream
            # Downcast to video_stream_profile and fetch intrinsics
            profile_depth = profile.get_stream(rs.stream.depth)
            intr_depth = profile_depth.as_video_stream_profile()
            intr_depth = intr_depth.get_intrinsics()

            # Extrinsic matrix from color sensor to Depth sensor ---------------
            profile_vid = profile_color.as_video_stream_profile()
            extr = profile_vid.get_extrinsics_to(profile_depth)
            extr_mat = np.eye(4)
            extr_mat[:3, :3] = np.array(extr.rotation).reshape(3, 3)
            extr_mat[:3, 3] = extr.translation

            # Depth scale ------------------------------------------------------
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            # print("Depth Scale is: ", depth_scale)

            # Write calibration data to json file ------------------------------
            calib_data = {}
            calib_data['color'] = []
            calib_data['color'].append({
                'width': intr_color.width,
                'height': intr_color.height,
                'intrinsic_mat': [intr_color.fx, 0, intr_color.ppx,
                                  0, intr_color.fy, intr_color.ppy,
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
            calib_data['T_color_depth'] = []
            calib_data['T_color_depth'].append({
                'rotation': extr.rotation,
                'translation': extr.translation
            })

            self.calib_data[dev_sn] = calib_data

            assert os.path.exists(storage_paths[dev_sn].calib)
            if os.path.isfile(storage_paths[dev_sn].calib):
                with open(storage_paths[dev_sn].calib, 'w') as outfile:
                    json.dump(calib_data, outfile, indent=4)
            else:
                filename = f'dev{dev_sn}_calib.txt'
                path = os.path.join(storage_paths[dev_sn].calib, filename)
                with open(path, 'w') as outfile:
                    json.dump(calib_data, outfile, indent=4)
