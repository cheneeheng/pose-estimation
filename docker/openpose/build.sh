#! /bin/sh

IMAGE_NAME="cheneeheng/openpose"

if [ $# -ge 1 ]; then

    if [ "$1" = "ubuntu20cpu" ]; then
        DOCKER_FILE="dockerfiles/Dockerfile.U20CPU"
        BASE_IMAGE="ubuntu:20.04"
        IMAGE_NAME="${IMAGE_NAME}:ubuntu20.04"
    elif [ "$1" = "ubuntu18" ]; then
        DOCKER_FILE="dockerfiles/Dockerfile.U18"
        BASE_IMAGE="nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04"
        IMAGE_NAME="${IMAGE_NAME}:cuda10.2-cudnn7-devel-ubuntu18.04"
    elif [ "$1" = "ubuntu20cu1171" ]; then
        DOCKER_FILE="dockerfiles/Dockerfile.U20"
        BASE_IMAGE="nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04"
        IMAGE_NAME="${IMAGE_NAME}:cuda11.7.1-cudnn8-devel-ubuntu20.04"
    elif [ "$1" = "ubuntu20cu1211" ]; then
        DOCKER_FILE="dockerfiles/Dockerfile.U20"
        BASE_IMAGE="nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04"
        IMAGE_NAME="${IMAGE_NAME}:cuda12.1.1-cudnn8-devel-ubuntu20.04"
    elif [ "$1" = "ubuntu20cu1171demo" ]; then
        DOCKER_FILE="dockerfiles/Dockerfile.U20DEMO"
        BASE_IMAGE="nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04"
        IMAGE_NAME="${IMAGE_NAME}:cuda11.7.1-cudnn8-devel-ubuntu20.04-demo"
    else
        echo "\033[0;31mUnknown argument, should be {ubuntu20cpu/ubuntu18/ubuntu20cu1171/ubuntu20cu1211/ubuntu20cu1171demo}\033[0m"
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

    echo "\033[0;31m1 argument is expected : {ubuntu20cpu/ubuntu18/ubuntu20cu1171/ubuntu20cu1211/ubuntu20cu1171demo}\033[0m"
    exit 1

fi
