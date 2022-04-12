#! /bin/sh

# This script builds docker image of the latest librealsense github tag
# Get the latest git TAG version
LIBRS_VERSION="2.50.0"
TARGET_TAG="librealsense:${LIBRS_VERSION}"

echo "Building images for librealsense version ${LIBRS_VERSION}"
DOCKER_BUILDKIT=1 docker build \
    --file Dockerfile \
    --target librealsense \
    --build-arg LIBRS_VERSION=${LIBRS_VERSION} \
    --build-arg UNAME_ARG=$1 \
    --tag ${TARGET_TAG} \
    .
echo "Built images for librealsense version ${LIBRS_VERSION}"
