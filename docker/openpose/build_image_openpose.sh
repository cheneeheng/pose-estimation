#! /bin/sh

IMAGE_NAME="openpose:cuda10.2-cudnn7-devel-ubuntu18.04"

echo "Building image : ${IMAGE_NAME}"
DOCKER_BUILDKIT=1 docker build \
    --file Dockerfile.OpenPose \
    --build-arg UNAME_ARG=$1 \
    --build-arg UID_ARG=$2 \
    --tag ${IMAGE_NAME} \
    .
echo "Built image : ${IMAGE_NAME}\n"
