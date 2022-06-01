#! /bin/sh

LIBRS_VERSION="2.50.0"

if [ $# -eq 1 ] || [ $# -eq 2 ]; then

    if [ "$1" = "ubuntu18" ]; then
        IMAGE_NAME="librealsense:ubuntu18.04-v${LIBRS_VERSION}"
        DOCKER_FILE="dockerfiles/Dockerfile.U18Realsense"
    elif [ "$1" = "ubuntu20" ]; then
        IMAGE_NAME="librealsense:ubuntu20.04-v${LIBRS_VERSION}"
        DOCKER_FILE="dockerfiles/Dockerfile.U20Realsense"
    else
        echo "Unknown argument, should be {ubuntu18/ubuntu20}"
        exit 1
    fi

    if [ $# -eq 1 ]; then
        echo "Building image : ${IMAGE_NAME}"
        DOCKER_BUILDKIT=1 docker build \
            --file ${DOCKER_FILE} \
            --target librealsense \
            --tag ${IMAGE_NAME} \
            .
        echo "Built image : ${IMAGE_NAME}\n"
    else
        IMAGE_NAME="${2%:*}-${IMAGE_NAME}-v${LIBRS_VERSION}"
        echo "Building image : ${IMAGE_NAME}"
        DOCKER_BUILDKIT=1 docker build \
            --file ${DOCKER_FILE} \
            --target librealsense \
            --build-arg BASE_IMAGE=$2 \
            --tag "${IMAGE_NAME}" \
            .
        echo "Built image : ${IMAGE_NAME}\n"

    fi

else

    echo "1 or 2 arguments are expected : {ubuntu18/ubuntu20}, {BASE_IMAGE}"
    exit 1

fi
