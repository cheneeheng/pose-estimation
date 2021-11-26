#! /bin/sh

# This script builds docker image of the latest librealsense github tag
# Get the latest git TAG version
LIBRS_VERSION="2.50.0"

echo "Building images for librealsense version ${LIBRS_VERSION}"
DOCKER_BUILDKIT=1 docker build \
    --file Dockerfile.RealsenseOnly \
    --target librealsense \
    --build-arg LIBRS_VERSION=${LIBRS_VERSION} \
    --tag librealsense:${LIBRS_VERSION} \
    .
echo "Built images for librealsense version ${LIBRS_VERSION}"
