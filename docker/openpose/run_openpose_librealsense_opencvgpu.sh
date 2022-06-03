#! /bin/sh

# This script builds docker image of the latest librealsense github tag
# Get the latest git TAG version
LIBRS_VERSION="2.50.0"
BASE_IMAGE_TAG="cuda11.5.2-cudnn8-devel-ubuntu20.04"
TARGET_TAG="openpose-librealsense-opencvgpu-user:${BASE_IMAGE_TAG}-v${LIBRS_VERSION}"

# DATA_PATH="/mnt/DHM-ICUSUITE-DS1/data/testing"
# DATA_PATH="/home/dhm/workspace/demo_event/data/openpose"
# CODE_PATH="/home/dhm/workspace/demo_event/code/pose-estimation"

DATA_PATH=$2
CODE_PATH=$3

# By using --device-cgroup-rule flag we grant the docker continer permissions -
# to the camera and usb endpoints of the machine.
# It also mounts the /dev directory of the host platform on the contianer
docker run -it --rm --gpus=all \
    -v /dev:/dev \
    --env DISPLAY=:3 \
    --env QT_X11_NO_MITSHM=1 \
    --env PYTHONPATH=/usr/local/lib:/code/pose-estimation:/code/pose-estimation/openpose:/code/pose-estimation/openpose/native/wrapper/2s_agcn \
    --device-cgroup-rule "c 81:* rmw" \
    --device-cgroup-rule "c 189:* rmw" \
    --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix \
    --mount type=bind,source=${CODE_PATH},target=/code/pose-estimation \
    --mount type=bind,source=${DATA_PATH},target=/data/openpose \
    ${TARGET_TAG} $1
