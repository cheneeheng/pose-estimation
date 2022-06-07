import cv2
import json
import numpy as np
import os

try:
    import pyrealsense2 as rs
except ImportError as e:
    print('Error: OpenPose library could not be found.')
    raise e

from typing import Optional, Type

from realsense.realsense_device_manager import Device
from realsense.realsense_device_manager import enumerate_connected_devices
from realsense.realsense_device_manager import post_process_depth_frame


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


class StoragePaths:
    def __init__(self):
        self.calib = None
        self.color = None
        self.depth = None
        self.timestamp = None
        self.timestamp_file = None


class StreamConfig:
    def __init__(self):
        self.fps = 30
        self.height = 480
        self.width = 848

    @property
    def data(self) -> dict:
        return {
            'width': self.width,
            'height': self.height,
            'framerate': self.fps
        }


class RealsenseWrapper:
    """Wrapper to run multiple realsense cameras.

    Code is written based on "realsense_device_manager.py" . Currently the
    code supports only reading depth and color images. Reading of IR stream
    is not implemented.

    """

    def __init__(self,
                 storage_paths_fn: Optional[Type[StoragePaths]] = None) -> None:
        super().__init__()
        # device data
        self.available_devices = enumerate_connected_devices(rs.context())
        self.enabled_devices = {}  # serial numbers of enabled devices
        self.calib_data = {}
        # rs align method
        self._align = rs.align(rs.stream.color)  # align depth to color frame
        # configurations
        self._rs_cfg = {}
        self.stream_config = StreamConfig()
        if storage_paths_fn is not None:
            self.storage_paths_per_dev = {sn: storage_paths_fn(sn)
                                          for sn, _ in self.available_devices}
        else:
            self.storage_paths_per_dev = {}

    def configure_stream(self, device_sn: Optional[str] = None) -> None:
        """Defines per device stream configurations.

        device('001622070408')
        device('001622070717')

        Args:
            device_sn (str, optional): serial number. Defaults to None.
        """
        if device_sn is not None:
            cfg = rs.config()
            cfg.enable_stream(stream_type=rs.stream.depth,
                              format=rs.format.z16,
                              **self.stream_config.data)
            cfg.enable_stream(stream_type=rs.stream.color,
                              format=rs.format.bgr8,
                              **self.stream_config.data)
            self._rs_cfg[device_sn] = cfg

    def initialize(self, enable_ir_emitter: bool = True) -> None:
        """Initializes the device pipelines and starts them.

        Args:
            enable_ir_emitter (bool, optional): Enable the IR for beter
                depth quality. Defaults to True.
        """
        if len(self._rs_cfg) == 0:
            self.configure_stream('default')

        for device_serial, product_line in self.available_devices:
            # Pipeline
            pipeline = rs.pipeline()
            cfg = self._rs_cfg.get(device_serial, self._rs_cfg['default'])
            cfg.enable_device(device_serial)
            pipeline_profile = pipeline.start(cfg)
            # IR for depth
            depth_sensor = pipeline_profile.get_device().first_depth_sensor()
            if enable_ir_emitter:
                if depth_sensor.supports(rs.option.emitter_enabled):
                    depth_sensor.set_option(rs.option.emitter_enabled,
                                            1 if enable_ir_emitter else 0)
                    # depth_sensor.set_option(rs.option.laser_power, 330)
            # Stored the enabled devices
            self.enabled_devices[device_serial] = (
                Device(pipeline, pipeline_profile, product_line))

    def set_storage_paths(self, paths: StoragePaths) -> None:
        self.storage_paths_per_dev = {sn: paths(sn)
                                      for sn in self.enabled_devices}

    def set_ir_laser_power(self, power: int = 300):
        """Sets the power of the IR laser. If power value is too high the
        rs connection will crash.

        https://github.com/IntelRealSense/librealsense/issues/1258

        Args:
            power (int, optional): IR power. Defaults to 300.
        """
        for _, dev in self.enabled_devices:
            sensor = dev.pipeline_profile.get_device().first_depth_sensor()
            if sensor.supports(rs.option.emitter_enabled):
                ir_range = sensor.get_option_range(rs.option.laser_power)
                if power + 10 > ir_range.max:
                    sensor.set_option(rs.option.laser_power, ir_range.max)
                else:
                    sensor.set_option(rs.option.laser_power, power + 10)

    def run(self, display: int = 0) -> dict:
        """Gets the frames streamed from the enabled rs devices.

        Args:
            display (int, optional): Whether to display the retrieved frames.
                The value corresponds to the scale to visualize the frames.
                Defaults to 0 = no display.

        Returns:
            dict: Empty dict or {serial_number: {data_type: data}}.
                data_type = color, depth, timestamp, calib
        """
        if len(self.enabled_devices) == 0:
            print("No devices are enabled...")
            return {}

        frames = {}
        while len(frames) < len(self.enabled_devices.items()):

            for dev_sn, dev in self.enabled_devices.items():

                storage_paths = self.storage_paths_per_dev.get(dev_sn, None)

                streams = dev.pipeline_profile.get_streams()
                frameset = dev.pipeline.poll_for_frames()

                if frameset.size() == len(streams):
                    frames[dev_sn] = {}

                    frames[dev_sn]['calib'] = self.calib_data[dev_sn]

                    timestamp = frameset.get_frame_metadata(
                        rs.frame_metadata_value.sensor_timestamp)
                    frames[dev_sn]['timestamp'] = timestamp

                    if storage_paths is not None:
                        ts_file = storage_paths.timestamp_file
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
                            frame_data = np.asanyarray(frame.get_data())
                            frames[dev_sn]['color'] = frame_data
                            if storage_paths is not None:
                                filepath = storage_paths.color
                                if filepath is not None:
                                    np.save(
                                        os.path.join(filepath, f'{timestamp}'),
                                        frame_data)
                        elif st == rs.stream.depth:
                            frame = aligned_frameset.first_or_default(st)
                            frame = post_process_depth_frame(frame)
                            frame_data = np.asanyarray(frame.get_data())
                            frames[dev_sn]['depth'] = frame_data
                            if storage_paths is not None:
                                filepath = storage_paths.depth
                                if filepath is not None:
                                    np.save(
                                        os.path.join(filepath, f'{timestamp}'),
                                        frame_data)

            if display > 0:
                if self._display_rs_data(frames, display):
                    return {}

            return frames

    def stop(self) -> None:
        """Stops the devices. """
        if len(self.enabled_devices) == 0:
            print("No devices are enabled...")
        else:
            for _, dev in self.enabled_devices.items():
                dev.pipeline.stop()

    def _display_rs_data(self, frames: dict, scale: int) -> bool:
        terminate = False
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
            images = cv2.resize(images, (images.shape[1]//scale,
                                         images.shape[0]//scale))
            cv2.namedWindow(f'{dev_sn}', cv2.WINDOW_AUTOSIZE)
            cv2.imshow(f'{dev_sn}', images)
            key = cv2.waitKey(30)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                cv2.waitKey(5)
                terminate = True

        return terminate

    def save_calibration(self) -> None:
        """Saves camera calibration. """

        if len(self.enabled_devices) == 0:
            print("No devices are enabled...")
            return

        for dev_sn, dev in self.enabled_devices.items():

            storage_paths = self.storage_paths_per_dev.get(dev_sn, None)
            if storage_paths is None:
                continue

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

            assert os.path.exists(storage_paths.calib)
            if os.path.isfile(storage_paths.calib):
                with open(storage_paths.calib, 'w') as outfile:
                    json.dump(calib_data, outfile, indent=4)
            else:
                filename = f'dev{dev_sn}_calib.txt'
                path = os.path.join(storage_paths.calib, filename)
                with open(path, 'w') as outfile:
                    json.dump(calib_data, outfile, indent=4)
