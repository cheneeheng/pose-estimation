FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

RUN apt-get update -y
RUN apt-get install -y sudo nano unzip wget curl git 

ENV TZ=Europe/Berlin
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

# ------------------------------------------------------------------------------
# Python 3.8
RUN apt-get install -y python3.8 python3.8-dev
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN ln -s /usr/bin/python3.8 /usr/bin/python3
RUN apt-get install -y python3-pip
## Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
## Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN pip install numpy==1.19.5

# ------------------------------------------------------------------------------
# OpenCV Apts
RUN apt-get install -y \
	ffmpeg \
	net-tools \
	libjpeg-dev \
	libpng-dev \
	libtiff-dev \
	libavcodec-dev \
	libavformat-dev\
	libswscale-dev \
	libavresample-dev \
	libgstreamer1.0-dev \
	libgstreamer-plugins-base1.0-dev \
	libxvidcore-dev \
	x264 \
	libx264-dev \
	libfaac-dev \
	libmp3lame-dev \
	libtheora-dev \
	libfaac-dev \
	libmp3lame-dev \
	libvorbis-dev \
	libopencore-amrnb-dev \
	libopencore-amrwb-dev

RUN apt-get install -y \
	libdc1394-22 \
	libdc1394-22-dev \
	libxine2-dev libv4l-dev \
	v4l-utils
RUN cd /usr/include/linux
RUN ln -s -f ../libv4l1-videodev.h videodev.h
RUN cd -

# OpenCV optionals
RUN apt-get install -y \
	libgtk-3-dev \
	libtbb-dev \
	libatlas-base-dev \
	gfortran \
	libeigen3-dev\
	intel-mkl
# RUN apt-get install libopenblas-base -y (USE CUBLAS)
# RUN apt-get install python3-vtk7 -y # (IP ERROR)

# ------------------------------------------------------------------------------
# Compiling OpenCV
RUN mkdir -p /usr/local/src/opencv
WORKDIR /usr/local/src/opencv
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.5.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.5.zip
RUN unzip opencv.zip
RUN unzip opencv_contrib.zip
RUN mv opencv-4.5.5 opencv
RUN mv opencv_contrib-4.5.5 opencv_contrib
RUN mkdir -p /usr/local/src/opencv/opencv/build
WORKDIR /usr/local/src/opencv/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D BUILD_EXAMPLES=ON \
	-D ENABLE_FAST_MATH=ON \
	-D MKL_LAPACKE_INCLUDE_DIR=/usr/include/mkl \
	-D MKL_ROOT_DIR=/usr/lib/x86_64-linux-gnu/mkl \ 
	-D MKL_WITH_TBB=ON \
	-D MKL_USE_SINGLE_DYNAMIC_LIBRARY=ON \ 
	-D OPENCV_DNN_CUDA=ON \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D OPENCV_EXTRA_MODULES_PATH=/usr/local/src/opencv/opencv_contrib/modules \
	# -D OPENCV_PYTHON3_VERSION=ON \
	-D CUDA_FAST_MATH=ON \
	-D WITH_CUBLAS=ON \
	-D WITH_CUDA=ON \
	-D WITH_CUDNN=ON \
	-D WITH_EIGEN=ON \
	-D WITH_TBB=ON \
	-D WITH_LIBREALSENSE=ON \
	-D WITH_V4L=ON \
	-D WITH_QT=OFF \
	# -D WITH_OPENGL=ON \
	-D WITH_GSTREAMER=ON \
	-D PYTHON_EXECUTABLE=/usr/bin/python \
	.. 2>&1 | tee output.log
RUN make -j8
RUN make install 
RUN ldconfig

WORKDIR /

# ------------------------------------------------------------------------------
# Install pip requirements
# RUN pip install tensorflow==2.5.0
# COPY requirements.txt /tmp
# RUN python -m pip install -r /tmp/requirements.txt

# ------------------------------------------------------------------------------
# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u ${UID_ARG} --disabled-password --gecos "" ${UNAME_ARG}
RUN echo "${UNAME_ARG}:${UNAME_ARG}" | chpasswd
RUN usermod -aG sudo ${UNAME_ARG}
USER ${UNAME_ARG}

#WORKDIR /code
