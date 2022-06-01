#! /bin/sh

if [ $# -ne 3 ]; then
    echo "3 arguments are expected : uname, uid, {Ubuntu18/Ubuntu20}"
    exit 1

else

    if [ "$3" = "Ubuntu18" ]; then
        IMAGE_NAME="openpose:cuda10.2-cudnn7-devel-ubuntu18.04"
        DOCKER_FILE="dockerfiles/Dockerfile.Ubuntu18Openpose"
    elif [ "$3" = "Ubuntu20" ]; then
        IMAGE_NAME="openpose:cuda11.5.2-cudnn8-devel-ubuntu20.04"
        DOCKER_FILE="dockerfiles/Dockerfile.Ubuntu20Openpose"
    else
        echo "Unknown 3rd argument, should be {Ubuntu18/Ubuntu20}"
        exit 1
    fi

    echo "Building image : $3"
    DOCKER_BUILDKIT=1 docker build \
        --file ${DOCKER_FILE} \
        --build-arg UNAME_ARG=$1 \
        --build-arg UID_ARG=$2 \
        --tag ${IMAGE_NAME} \
        .
    echo "Built image : ${IMAGE_NAME}\n"

fi
