import os
import time
import sys

from rs_py.rs_run_devices import get_rs_parser
from rs_py.rs_run_devices import printout
from rs_py.rs_run_devices import RealsenseWrapper


if __name__ == "__main__":
    args, remain_args = get_rs_parser().parse_known_args()
    args.rs_steps = 10
    args.rs_fps = 30
    args.rs_display_frame = 1
    args.rs_save_with_key = False
    args.rs_save_data = False
    args.rs_save_path = ''
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

    rsw.flush_frames(args.rs_fps * 5)
    time.sleep(3)

    device_sn = list(rsw.enabled_devices.keys())[0]

    try:
        c = 0
        max_c = int(1e8)

        while True:

            rsw.step(
                display=args.rs_display_frame,
                display_and_save_with_key=args.rs_save_with_key
            )

            # print(rsw.frames[device_sn]['color_framedata'].shape)
            # print(rsw.frames[device_sn]['depth_framedata'].shape)

            if rsw.key & 0xFF == ord('q'):
                break

            if c % args.rs_fps == 0:
                printout(
                    f"Step {c:12d} :: "
                    f"{[i.get('color_timestamp', None) for i in rsw.frames.values()]} :: "  # noqa
                    f"{[i.get('depth_timestamp', None) for i in rsw.frames.values()]}",  # noqa
                    'i'
                )

            if not len(rsw.frames) > 0:
                printout(f"Empty...", 'w')
                continue

            c += 1
            if c > args.rs_fps * args.rs_steps or c > max_c:
                break

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        printout(f"{e}", 'e')
        printout(f"Stopping RealSense devices...", 'i')
        rsw.stop()

    finally:
        printout(f"Final RealSense devices...", 'i')
        rsw.stop()

    printout(f"Finished...", 'i')
