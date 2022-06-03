from .wrapper.skeleton import PyOpenPoseNative
from .wrapper.skeleton import get_3d_skeleton

try:
    from .wrapper.extract_3D_skeleton_from_rs import OpenposeStoragePaths
    from .wrapper.extract_3D_skeleton_from_rs import get_parser as get_op_parser
except ImportError:
    print("Warning: pyrealsense is not found")
