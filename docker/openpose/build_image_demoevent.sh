#! /bin/sh

# This script builds docker image of the latest librealsense github tag
# Get the latest git TAG version
TARGET_TAG="openpose-librealsense-demoevent:cuda10.2-cudnn7-devel-ubuntu18.04-2.50.0"

echo "Building image for demo event"
DOCKER_BUILDKIT=1 docker build \
    --file Dockerfile.DemoEvent \
    --build-arg UNAME_ARG=$1 \
    --tag ${TARGET_TAG} \
    .
echo "Built image for demo event"
