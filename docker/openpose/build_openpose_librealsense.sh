#! /bin/sh

# OPENPOSE
sh build.sh ubuntu20

# REALSENSE
P=$(pwd)
mkdir tmp
cd tmp
git clone https://github.com/cheneeheng/realsense-simple-wrapper.git
cd realsense-simple-wrapper/docker/realsense
sh build.sh ubuntu20full openpose:cuda11.5.2-cudnn8-devel-ubuntu20.04
cd $P
rm -rf tmp

# ADD USER
mkdir tmp
cp -rf ../_user tmp
cd tmp/_user
sh build.sh dhm 1001 openpose-librealsense-full:cuda11.5.2-cudnn8-devel-ubuntu20.04-v2.50.0
cd $P
rm -rf tmp
