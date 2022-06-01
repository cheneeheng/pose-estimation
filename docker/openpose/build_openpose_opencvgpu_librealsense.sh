#! /bin/sh

# OPENPOSE
sh build.sh ubuntu20

mkdir tmp

# REALSENSE
cp -rf ../realsense tmp
cd tmp/realsense
sh build.sh ubuntu20 openpose:cuda11.5.2-cudnn8-devel-ubuntu20.04
cd ../..

# OPENCV GPU
cp -rf ../opencv_gpu tmp
cd tmp/opencv_gpu
sh build.sh ubuntu20 openpose-opencvgpu:cuda11.5.2-cudnn8-devel-ubuntu20.04
cd ../..

# ADD USER
cp -rf ../_user tmp
cd tmp/_user
sh build.sh ubuntu20 openpose-opencvgpu-librealsense:cuda11.5.2-cudnn8-devel-ubuntu20.04-v2.50.0
cd ../..

rm -rf tmp
