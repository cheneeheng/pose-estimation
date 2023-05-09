try:
    from .op_py.archiv.extract_3D_skeleton_from_rs import OpenposeStoragePaths
    from .op_py.archiv.extract_3D_skeleton_from_rs import get_parser as get_op_parser  # noqa
except ImportError:
    print("Warning: pyrealsense is not found")
