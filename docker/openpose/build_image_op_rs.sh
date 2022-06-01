#! /bin/sh

# This script builds docker image of the latest librealsense github tag
# Get the latest git TAG version
LIBRS_VERSION="2.50.0"

if [ $# -ne 3 ]; then
    echo "3 arguments are expected : uname, uid, {Ubuntu18/Ubuntu20}"
    exit 1

else

    if [ "$3" = "Ubuntu18" ]; then
        TARGET_TAG="openpose-librealsense:cuda10.2-cudnn7-devel-ubuntu18.04-${LIBRS_VERSION}"
        DOCKER_FILE="dockerfiles/Dockerfile.Ubuntu18OpenposeRealsense"
    elif [ "$3" = "Ubuntu20" ]; then
        TARGET_TAG="openpose-librealsense:cuda11.5.2-cudnn8-devel-ubuntu20.04-${LIBRS_VERSION}"
        DOCKER_FILE="dockerfiles/Dockerfile.Ubuntu20OpenposeRealsense"
    else
        echo "Unknown 3rd argument, should be {Ubuntu18/Ubuntu20}"
        exit 1
    fi

    echo "Building images for librealsense version ${LIBRS_VERSION} with openpose"
    DOCKER_BUILDKIT=1 docker build \
        --file ${DOCKER_FILE} \
        --target openpose-librealsense \
        --build-arg LIBRS_VERSION=${LIBRS_VERSION} \
        --build-arg UNAME_ARG=$1 \
        --build-arg UID_ARG=$2 \
        --tag ${TARGET_TAG} \
        .
    echo "Built images for librealsense version ${LIBRS_VERSION} with openpose"

fi
