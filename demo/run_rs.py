from rs_py.rs_run_devices import get_rs_parser
from rs_py.rs_run_devices import get_parser
from rs_py.rs_run_devices import run_devices


if __name__ == "__main__":
    args, remain_args = get_rs_parser().parse_known_args()
    args.rs_steps = 10
    args.rs_fps = 30
    print("========================================")
    print(">>>>> args <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")

    args_local, _ = get_parser().parse_known_args(remain_args)
    if args_local.rs_test_init_runtime:
        raise ValueError("Error")
    elif args_local.rs_test_hardware_reset_runtime:
        raise ValueError("Error")
    else:
        run_devices(args)
