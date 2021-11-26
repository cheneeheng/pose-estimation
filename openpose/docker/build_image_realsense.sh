#! /bin/sh

# This script builds docker image of the latest librealsense github tag
# Get the latest git TAG version
LIBRS_VERSION="2.50.0"
OPENPOSE_TAG="cuda10.2-cudnn7-devel-ubuntu18.04"
TARGET_TAG="openpose-librealsense:${OPENPOSE_TAG}-${LIBRS_VERSION}"

echo "Building images for librealsense version ${LIBRS_VERSION} with openpose"
DOCKER_BUILDKIT=1 docker build \
    --file Dockerfile.Realsense \
    --target openpose-librealsense \
    --build-arg LIBRS_VERSION=${LIBRS_VERSION} \
    --build-arg OPENPOSE_IMAGE_ARG="openpose:${OPENPOSE_TAG}" \
    --build-arg UNAME_ARG=$1 \
    --tag ${TARGET_TAG} \
    .
echo "Built images for librealsense version ${LIBRS_VERSION} with openpose"
