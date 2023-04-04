#! /bin/sh

if [ $# -eq 4 ]; then

    IMAGE_NAME="openpose-user:cuda10.2-cudnn7-devel-ubuntu18.04"
    DATA_PATH=$2
    CODE_PATH=$3

    if [ "$1" = "ubuntu18" ]; then
        IMAGE_NAME="openpose:cuda10.2-cudnn7-devel-ubuntu18.04"
    elif [ "$1" = "ubuntu20" ]; then
        IMAGE_NAME="openpose:cuda11.7.1-cudnn8-devel-ubuntu20.04"
    elif [ "$1" = "ubuntu20cpu" ]; then
        IMAGE_NAME="openpose:ubuntu20.04"
    else
        echo "Unknown argument, should be {ubuntu18/ubuntu20/ubuntu20cpu}"
        exit 1
    fi

    # By using --device-cgroup-rule flag we grant the docker continer permissions -
    # to the camera and usb endpoints of the machine.
    # It also mounts the /dev directory of the host platform on the contianer
    docker run -it --rm --gpus=all \
        -v /dev:/dev \
        --env DISPLAY=:0 \
        --env QT_X11_NO_MITSHM=1 \
        --env PYTHONPATH=/usr/local/lib:/code/pose-estimation \
        --device-cgroup-rule "c 81:* rmw" \
        --device-cgroup-rule "c 189:* rmw" \
        --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix \
        --mount type=bind,source=${CODE_PATH},target=/code/pose-estimation \
        --mount type=bind,source=${DATA_PATH},target=/data/openpose \
        --workdir /code/pose-estimation \
        ${IMAGE_NAME} $4

else

    echo "4 arguments are expected : {ubuntu18/ubuntu20/ubuntu20cpu} {datapath} {codepath} {cmd}"
    exit 1

fi
