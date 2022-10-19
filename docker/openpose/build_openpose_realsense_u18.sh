#! /bin/bash

# builds an image with latest openpose + realsense + opencvgpu + (optional) add user
# 2 ARGS IN ORDER: {USERNAME}, {PID}

BASE_IMAGE="nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04"
DOCKER_FILE="dockerfiles/Dockerfile.U18Openpose"

BASE_NAME="cuda10.2-cudnn7-devel-ubuntu18.04"
OP_IMAGE_NAME="openpose:${BASE_NAME}"
RS_IMAGE_NAME="openpose-librealsense:${BASE_NAME}-v2.50.0"
CV_IMAGE_NAME="openpose-librealsense-opencvgpu:${BASE_NAME}-v2.50.0"

mkdir tmp
P=$(pwd)

# BUILD OPENPOSE IMAGE ---------------------------------------------------------
echo "Building image : ${OP_IMAGE_NAME}"
DOCKER_BUILDKIT=1 docker build \
    --file ${DOCKER_FILE} \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    --tag "${OP_IMAGE_NAME}" \
    .
echo "Built image : ${OP_IMAGE_NAME}\n"

# REALSENSE ----------------------------------------------------------------
echo "building realsense image...\n"
cd tmp
git clone https://github.com/cheneeheng/realsense-simple-wrapper.git
cd realsense-simple-wrapper/docker/realsense
sh build.sh ubuntu18 ${OP_IMAGE_NAME}
cd $P

# OPENCV GPU ---------------------------------------------------------------
echo "building opencvgpu image..."
cp -rf ../opencvgpu tmp
cd tmp/opencvgpu
sh build.sh ubuntu18 ${RS_IMAGE_NAME}
cd $P

# ADD USER ---------------------------------------------------------------------
if [ $# -gt 0 ]; then
    cp -rf ../_user tmp
    cd tmp/_user
    echo "adding user...\n"
    sh build.sh $1 $2 ${IMAGE_NAME}
    cd $P
fi

rm -rf tmp
