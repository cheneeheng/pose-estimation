import argparse
import os

from datetime import datetime
from typing import Type, Optional

from realsense import RealsenseWrapper
from realsense import StoragePaths


class RealsenseStoragePaths(StoragePaths):
    def __init__(self, device_sn: str = ''):
        super().__init__()
        base_path = '/data/realsense'
        date_time = datetime.now().strftime("%y%m%d%H%M%S")
        self.calib = f'{base_path}/calib/{date_time}_dev{device_sn}'
        self.color = f'{base_path}/color/{date_time}_dev{device_sn}'
        self.depth = f'{base_path}/depth/{date_time}_dev{device_sn}'
        self.timestamp = f'{base_path}/timestamp/{date_time}_dev{device_sn}'
        self.timestamp_file = os.path.join(self.timestamp, 'timestamp.txt')
        os.makedirs(self.calib, exist_ok=True)
        os.makedirs(self.color, exist_ok=True)
        os.makedirs(self.depth, exist_ok=True)
        os.makedirs(self.timestamp, exist_ok=True)


def get_parser():
    parser = argparse.ArgumentParser(description='Run RealSense devices.')
    parser.add_argument('--rs-fps',
                        type=int,
                        default=30,
                        help='fps')
    parser.add_argument('--rs-image-width',
                        type=int,
                        default=848,
                        help='image width in px')
    parser.add_argument('--rs-image-height',
                        type=int,
                        default=480,
                        help='image height in px')
    parser.add_argument('--rs-laser-power',
                        type=int,
                        default=150,
                        help='laser power')
    parser.add_argument('--rs-display-frame',
                        type=int,
                        default=0,
                        help='scale for displaying realsense raw images.')
    parser.add_argument('--rs-save-data',
                        default=True,
                        help='if true, saves realsense frames.')
    parser.add_argument('--rs-use-one-dev-only',
                        type=bool,
                        default=False,
                        help='use 1 rs device only.')
    return parser


def initialize_rs_devices(
        arg: argparse.Namespace,
        storage_paths: Optional[Type[StoragePaths]] = RealsenseStoragePaths
) -> RealsenseWrapper:
    rsw = RealsenseWrapper(storage_paths if arg.rs_save_data else None)
    rsw.stream_config.fps = arg.rs_fps
    rsw.stream_config.height = arg.rs_image_height
    rsw.stream_config.width = arg.rs_image_width
    if arg.rs_use_one_dev_only:
        rsw.available_devices = rsw.available_devices[0:1]
    rsw.initialize()
    rsw.set_ir_laser_power(arg.rs_laser_power)
    rsw.save_calibration()
    print("Initialized RealSense devices...")
    return rsw


if __name__ == "__main__":
    arg = get_parser().parse_args()
    rsw = initialize_rs_devices(arg)
    print("Starting frame capture loop...")
    try:
        while True:
            print("Running...")
            frames = rsw.run(display=arg.rs_display_frame)
            if not len(frames) > 0:
                continue
    except:  # noqa
        print("Stopping RealSense devices...")
        rsw.stop()
    finally:
        rsw.stop()
