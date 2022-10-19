#! /bin/bash

# builds an image with latest openpose + realsense 2.25 + opencvgpu + add user
# this image uses CU11+CUDNN8

if [ $# -le 4 ] && [ $# -gt 0 ]; then

    mkdir tmp
    P=$(pwd)

    # OPENPOSE -----------------------------------------------------------------
    X=0
    for var in "$@"; do
        if [ "$var" = "openpose" ]; then
            X=1
        fi
    done
    if [ $X = 1 ]; then
        echo "building openpose image..."
        sh build_openpose.sh ubuntu20
    else
        echo "{openpose} argument is required for this script..."
        exit 1
    fi

    # REALSENSE ----------------------------------------------------------------
    X=0
    for var in "$@"; do
        if [ "$var" = "realsense" ]; then
            X=1
        fi
    done
    if [ $X = 1 ]; then
        echo "building realsense image..."
        cd tmp
        git clone https://github.com/cheneeheng/realsense-simple-wrapper.git
        cd realsense-simple-wrapper/docker/realsense
        sh build.sh ubuntu20full openpose:cuda11.5.2-cudnn8-devel-ubuntu20.04
        cd $P
    fi

    # OPENCV GPU ---------------------------------------------------------------
    X=0
    for var in "$@"; do
        if [ "$var" = "opencvgpu" ]; then
            X=1
        fi
    done
    if [ $X = 1 ]; then
        echo "building opencvgpu image..."
        cp -rf ../opencvgpu tmp
        cd tmp/opencvgpu
        sh build.sh ubuntu20 openpose-librealsense:cuda11.5.2-cudnn8-devel-ubuntu20.04-v2.50.0
        cd $P
    fi

    # ADD USER -----------------------------------------------------------------
    X=0
    for var in "$@"; do
        if [ "$var" = "user" ]; then
            X=1
        fi
    done
    if [ $X = 1 ]; then
        echo "building opencvgpu image..."
        cp -rf ../_user tmp
        cd tmp/_user
        sh build.sh dhm 1001 openpose-librealsense-opencvgpu:cuda11.5.2-cudnn8-devel-ubuntu20.04-v2.50.0
        cd $P
    fi

    rm -rf tmp

else
    echo "1 - 4 arguments are expected : {openpose}, {realsense}, {opencvgpu}, {user}"
    exit 1

fi
