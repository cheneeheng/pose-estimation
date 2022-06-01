# RealSense in Python

## Files:

The code here uses the pyrealsense2 python package to run and stream images/frames from realsense devices.

| File                                                       | Details                                                                                             |
| :--------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- |
| [realsense_run_devices.py](realsense_run_devices.py)       | Runs X realsense devices that are connected to the PC.                                              |
| [realsense_wrapper.py](realsense_wrapper.py)               | Contains the _RealsenseWrapper_ class that contains all the functions to run the realsense devices. |
| [realsense_device_manager.py](realsense_device_manager.py) | Helper functions from the official realsense repo.                                                  |
|                                                            |                                                                                                     |

## Good to know infos:

### Sensor timestamp [link](https://github.com/IntelRealSense/librealsense/issues/2188)
- SENSOR_TIMESTAMP: Device clock / sensor starts taking the image
- FRAME_TIMESTAMP: Device clock / frame starts to transfer to the driver
- BACKEND_TIMESTAMP: PC clock / frame starts to transfer to the driver
- TIME_OF_ARRIVAL: PC clock / frame is transfered to the driver

### HW Sync [link](https://dev.intelrealsense.com/docs/external-synchronization-of-intel-realsense-depth-cameras)
The realsense can be syncroized using an external sync cable so that the cameras captures at the exact same timestamp.


