import os
import time
import traceback
import sys

from rs_py.rs_run_devices import get_rs_parser
from rs_py.rs_run_devices import printout
from rs_py.rs_run_devices import RealsenseWrapper


if __name__ == "__main__":
    args, remain_args = get_rs_parser().parse_known_args()
    args.rs_steps = 60
    args.rs_fps = 30
    args.rs_image_width = 848
    args.rs_image_height = 480
    # args.rs_color_format = rs.format.bgr8
    # args.rs_depth_format = rs.format.z16
    args.rs_display_frame = 1
    args.rs_save_with_key = False
    args.rs_save_data = True
    args.rs_save_path = '/data/realsense'
    args.rs_use_one_dev_only = False
    args.rs_dev = None
    args.rs_ip = None
    args.rs_verbose = False
    args.rs_autoexposure = True
    args.rs_depth_sensor_autoexposure_limit = 200000.0
    args.rs_enable_ir_emitter = True
    args.rs_ir_emitter_power = 290

    print("========================================")
    print(">>>>> args <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")

    rsw = RealsenseWrapper(args, args.rs_dev)
    rsw.initialize_depth_sensor_ae()

    if len(rsw.enabled_devices) == 0:
        raise ValueError("no devices connected")

    rsw = RealsenseWrapper(args, args.rs_dev)
    rsw.initialize()

    if args.rs_save_data:
        rsw.storage_paths.create()
        rsw.save_calib()

    rsw.flush_frames(args.rs_fps * 5)
    time.sleep(3)

    device_sn = list(rsw.enabled_devices.keys())[0]
    timer = []

    try:
        c = 0
        max_c = int(1e8)

        while True:

            if c % (args.rs_fps//1) == 0:
                display = args.rs_display_frame
                use_colorizer = True
            else:
                display = 0
                use_colorizer = False

            start = time.time()
            rsw.step(
                display=display,
                display_and_save_with_key=args.rs_save_with_key,
                use_colorizer=use_colorizer
            )
            timer.append(time.time() - start)

            # print(rsw.frames[device_sn]['color_framedata'].shape)
            # print(rsw.frames[device_sn]['depth_framedata'].shape)

            if rsw.key & 0xFF == ord('q'):
                break

            if c % args.rs_fps == 0:
                printout(
                    f"Step {c:12d} :: "
                    f"{len(timer)//sum(timer)} :: "
                    f"{[i.get('color_timestamp', None) for i in rsw.frames.values()]} :: "  # noqa
                    f"{[i.get('depth_timestamp', None) for i in rsw.frames.values()]}",  # noqa
                    'i'
                )
                timer = []

            if not len(rsw.frames) > 0:
                printout(f"Empty...", 'w')
                continue

            c += 1
            if c > args.rs_fps * args.rs_steps or c > max_c:
                break

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        printout(f"{exc_type}, {fname}, {exc_tb.tb_lineno}", 'e')
        printout(f"Exception msg : {e}", 'e')
        traceback.print_tb(exc_tb)
        printout(f"Stopping RealSense devices...", 'i')
        rsw.stop()

    finally:
        printout(f"Final RealSense devices...", 'i')
        rsw.stop()

    printout(f"Finished...", 'i')
