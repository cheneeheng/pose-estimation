#! /bin/sh

# This script builds docker image of the latest librealsense github tag
# Get the latest git TAG version
LIBRS_VERSION="2.50.0"
OPENPOSE_TAG="cuda10.2-cudnn7-devel-ubuntu18.04"
TARGET_TAG="openpose-librealsense:${OPENPOSE_TAG}-${LIBRS_VERSION}"

DATA_PATH="/home/dhm/workspace/demo_event/data/openpose"
CODE_PATH="/home/dhm/workspace/demo_event/code/pose-estimation/openpose"
AGCN_DATA_PATH="/home/dhm/workspace/demo_event/data/2s-agcn"
AGCN_CODE_PATH="/home/dhm/workspace/demo_event/code/2s-AGCN"

# By using --device-cgroup-rule flag we grant the docker continer permissions -
# to the camera and usb endpoints of the machine.
# It also mounts the /dev directory of the host platform on the contianer
docker run -it --rm --gpus=all \
    -v /dev:/dev \
    --env DISPLAY=:3 \
    --env QT_X11_NO_MITSHM=1 \
    --env PYTHONPATH=/usr/local/lib:/code/openpose/native/wrapper/2s_agcn \
    --device-cgroup-rule "c 81:* rmw" \
    --device-cgroup-rule "c 189:* rmw" \
    --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix \
    --mount type=bind,source=${CODE_PATH},target=/code/openpose \
    --mount type=bind,source=${DATA_PATH},target=/data/openpose \
    --mount type=bind,source=${AGCN_CODE_PATH},target=/code/openpose/native/wrapper/2s_agcn \
    --mount type=bind,source=${AGCN_DATA_PATH},target=/data/2s-agcn \
    ${TARGET_TAG} $1
