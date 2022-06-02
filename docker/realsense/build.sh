#! /bin/sh

LIBRS_VERSION="2.50.0"
IMAGE_NAME="librealsense"

if [ $# -eq 1 ] || [ $# -eq 2 ]; then

    if [ "$1" = "ubuntu18" ]; then
        DOCKER_FILE="dockerfiles/Dockerfile.U18Realsense"
    elif [ "$1" = "ubuntu20" ]; then
        DOCKER_FILE="dockerfiles/Dockerfile.U20Realsense"
    else
        echo "Unknown argument, should be {ubuntu18/ubuntu20}"
        exit 1
    fi

    if [ $# -eq 1 ]; then

        if [ "$1" = "ubuntu18" ]; then
            IMAGE_NAME="${IMAGE_NAME}:ubuntu18.04-v${LIBRS_VERSION}"
        elif [ "$1" = "ubuntu20" ]; then
            IMAGE_NAME="${IMAGE_NAME}:ubuntu20.04-v${LIBRS_VERSION}"
        fi

        echo "Building image : ${IMAGE_NAME}"
        DOCKER_BUILDKIT=1 docker build \
            --file ${DOCKER_FILE} \
            --target librealsense \
            --build-arg LIBRS_VERSION=${LIBRS_VERSION} \
            --tag ${IMAGE_NAME} \
            .
        echo "Built image : ${IMAGE_NAME}\n"

    else

        if [ "$1" = "ubuntu18" ]; then
            IMAGE_NAME="${2%:*}-${IMAGE_NAME}:${2#*:}-v${LIBRS_VERSION}"
        elif [ "$1" = "ubuntu20" ]; then
            IMAGE_NAME="${2%:*}-${IMAGE_NAME}:${2#*:}-v${LIBRS_VERSION}"
        fi

        echo "Building image : ${IMAGE_NAME}"
        DOCKER_BUILDKIT=1 docker build \
            --file ${DOCKER_FILE} \
            --target librealsense \
            --build-arg BASE_IMAGE=$2 \
            --build-arg LIBRS_VERSION=${LIBRS_VERSION} \
            --tag "${IMAGE_NAME}" \
            .
        echo "Built image : ${IMAGE_NAME}\n"

    fi

else

    echo "1 or 2 arguments are expected : {ubuntu18/ubuntu20}, {BASE_IMAGE}"
    exit 1

fi
