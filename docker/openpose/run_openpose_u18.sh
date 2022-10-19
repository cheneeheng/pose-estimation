#! /bin/sh

# THIS SCRIPT REQUIRES 3 ARGS: CMD + host DATA_PATH + host CODE_PATH

IMAGE_NAME="openpose-user:cuda10.2-cudnn7-devel-ubuntu18.04"
DATA_PATH=$2
CODE_PATH=$3

# By using --device-cgroup-rule flag we grant the docker continer permissions -
# to the camera and usb endpoints of the machine.
# It also mounts the /dev directory of the host platform on the contianer
docker run -it --rm --gpus=all \
    -v /dev:/dev \
    --env DISPLAY=:1 \
    --env QT_X11_NO_MITSHM=1 \
    --env PYTHONPATH=/usr/local/lib:/code/pose-estimation:/code/pose-estimation/openpose \
    --device-cgroup-rule "c 81:* rmw" \
    --device-cgroup-rule "c 189:* rmw" \
    --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix \
    --mount type=bind,source=${CODE_PATH},target=/code/pose-estimation \
    --mount type=bind,source=${DATA_PATH},target=/data/openpose \
    --workdir /code/pose-estimation \
    ${IMAGE_NAME} $1
