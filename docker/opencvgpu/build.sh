#! /bin/sh

IMAGE_NAME="opencvgpu"

if [ $# -eq 1 ] || [ $# -eq 2 ]; then

    if [ "$1" = "ubuntu20" ]; then
        DOCKER_FILE="dockerfiles/Dockerfile.U20Opencvgpu"
    else
        echo "Unknown argument, should be {ubuntu20}"
        exit 1
    fi

    if [ $# -eq 1 ]; then

        if [ "$1" = "ubuntu18" ]; then
            IMAGE_NAME="${IMAGE_NAME}:cuda10.2-cudnn7-devel-ubuntu18.04"
        elif [ "$1" = "ubuntu20" ]; then
            IMAGE_NAME="${IMAGE_NAME}:cuda11.5.2-cudnn8-devel-ubuntu20.04"
        fi

        echo "Building image : ${IMAGE_NAME}"
        DOCKER_BUILDKIT=1 docker build \
            --file ${DOCKER_FILE} \
            --tag ${IMAGE_NAME} \
            .
        echo "Built image : ${IMAGE_NAME}\n"

    else
        
        IMAGE_NAME="${2%:*}-${IMAGE_NAME}:${2#*:}"
        
        echo "Building image : ${IMAGE_NAME}"
        DOCKER_BUILDKIT=1 docker build \
            --file ${DOCKER_FILE} \
            --build-arg BASE_IMAGE=$2 \
            --tag "${IMAGE_NAME}" \
            .
        echo "Built image : ${IMAGE_NAME}\n"
    fi

else

    echo "1 or 2 arguments are expected : {ubuntu18/ubuntu20}, {BASE_IMAGE}"
    exit 1

fi
