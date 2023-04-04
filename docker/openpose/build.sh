#! /bin/sh

IMAGE_NAME="cheneeheng/openpose"

if [ $# -ge 1 ]; then

    if [ "$1" = "ubuntu18" ]; then
        DOCKER_FILE="dockerfiles/Dockerfile.U18"
        BASE_IMAGE="nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04"
        IMAGE_NAME="${IMAGE_NAME}:cuda10.2-cudnn7-devel-ubuntu18.04"
    elif [ "$1" = "ubuntu20" ]; then
        DOCKER_FILE="dockerfiles/Dockerfile.U20"
        BASE_IMAGE="nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04"
        IMAGE_NAME="${IMAGE_NAME}:cuda11.7.1-cudnn8-devel-ubuntu20.04"
    elif [ "$1" = "ubuntu20cpu" ]; then
        DOCKER_FILE="dockerfiles/Dockerfile.U20CPU"
        BASE_IMAGE="ubuntu:20.04"
        IMAGE_NAME="${IMAGE_NAME}:ubuntu20.04"
    else
        echo "Unknown argument, should be {ubuntu18/ubuntu20/ubuntu20cpu}"
        exit 1
    fi

    echo "Building image : ${IMAGE_NAME}"
    DOCKER_BUILDKIT=1 docker build \
        --file ${DOCKER_FILE} \
        --build-arg BASE_IMAGE=${BASE_IMAGE} \
        --tag ${IMAGE_NAME} \
        .
    echo "Built image : ${IMAGE_NAME}\n"

    # ADD USER ---------------------------------------------------------------------
    if [ $# -gt 1 ]; then
        mkdir tmp
        P=$(pwd)
        cp -rf ../_user tmp
        cd tmp/_user
        echo "adding user..."
        sh build.sh $1 $2 ${IMAGE_NAME}
        echo "added user...\n"
        cd $P
        rm -rf tmp
    fi


else

    echo "1 argument is expected : {ubuntu18/ubuntu20/ubuntu20cpu}"
    exit 1

fi
