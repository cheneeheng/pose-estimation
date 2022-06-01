#! /bin/sh

LIBRS_VERSION="2.50.0"

if [ $# -eq 1 ]; then

    if [ "$1" = "ubuntu18" ]; then
        TARGET_TAG="librealsense:v${LIBRS_VERSION}-ubuntu18.04"
    elif [ "$1" = "ubuntu20" ]; then
        TARGET_TAG="librealsense:v${LIBRS_VERSION}-ubuntu20.04"
    else
        echo "Unknown argument, should be {ubuntu18/ubuntu20}"
        exit 1
    fi

    # By using --device-cgroup-rule flag we grant the docker continer permissions -
    # to the camera and usb endpoints of the machine.
    # It also mounts the /dev directory of the host platform on the contianer

    docker run -it --rm \
        -v /dev:/dev \
        --device-cgroup-rule "c 81:* rmw" \
        --device-cgroup-rule "c 189:* rmw" \
        ${TARGET_TAG} $2

else

    echo "at least 1 argument is expected : {ubuntu18/ubuntu20}"
    exit 1

fi
