FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
# FROM opencv-4.5.2:cuda11.3.0-cudnn8-devel-ubuntu20.04
# ENV LANG en_US.UTF-8

RUN apt-get update -y
RUN apt-get install -y sudo nano unzip wget curl git 

ENV TZ=Europe/Berlin
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

# Python 3.8
RUN apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y python3.8 python3.8-dev python3.8-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
## Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
## Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Apts
RUN apt-get install -y libprotobuf-dev protobuf-compiler libgoogle-glog-dev
RUN apt-get install -y libboost-all-dev libhdf5-dev libatlas-base-dev
RUN apt-get install -y libopencv-dev

# pip
WORKDIR /tmp
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&  python3 get-pip.py
COPY requirements.txt /tmp
RUN python3 -m pip install -r /tmp/requirements.txt

# ------------------------------------------------------------------------------
# TAKEN FROM : https://github.com/esemeniuc/openpose-docker/blob/master/Dockerfile
# and : https://hub.docker.com/r/cwaffles/openpose

#replace cmake as old version has CUDA variable bugs
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1-Linux-x86_64.tar.gz && \
    tar xzf cmake-3.21.1-Linux-x86_64.tar.gz -C /opt && \
    rm cmake-3.21.1-Linux-x86_64.tar.gz
ENV PATH="/opt/cmake-3.21.1-linux-x86_64/bin:${PATH}"

WORKDIR /usr/local/src/openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git .
WORKDIR /usr/local/src/openpose/build
RUN cmake -D DOWNLOAD_BODY_25_MODEL=ON \
          -D DOWNLOAD_BODY_COCO_MODEL=ON \
          -D DOWNLOAD_BODY_MPI_MODEL=ON \
          -D DOWNLOAD_FACE_MODEL=OFF \
          -D DOWNLOAD_HAND_MODEL=OFF \ 
          -D CUDA_ARCH=Manual \
          -D CUDA_ARCH_BIN="5.3 6.0 6.1 7.0 7.5" \
          -D CUDA_ARCH_PTX="7.5" \ 
          -D BUILD_PYTHON=ON \
          -D PYTHON_EXECUTABLE="/usr/bin/python3.8" \
          -D PYTHON_LIBRARY="/usr/lib/x86_64-linux-gnu/libpython3.8m.so" ..
RUN make -j8

WORKDIR /usr/local/src/openpose/build/python/openpose
RUN make install

RUN cp ./pyopenpose.cpython-38-x86_64-linux-gnu.so /usr/local/lib/python3.8/dist-packages
RUN ln -s /usr/local/lib/python3.8/dist-packages/pyopenpose.cpython-38-x86_64-linux-gnu.so /usr/local/lib/python3.8/dist-packages/pyopenpose
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/src/openpose/build/python/openpose"

# ------------------------------------------------------------------------------

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 1000 --disabled-password --gecos "" chen
RUN echo "chen:chen" | chpasswd 
RUN usermod -aG sudo chen
USER chen
WORKDIR /home/chen


# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.177.0/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version: 3, 3.9, 3.8, 3.7, 3.6
#ARG VARIANT="3.9"
#FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# [Option] Install Node.js
#ARG INSTALL_NODE="true"
#ARG NODE_VERSION="lts/*"
#RUN if [ "${INSTALL_NODE}" = "true" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# [Optional] Uncomment this line to install global node packages.
# RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && npm install -g <your-package-here>" 2>&1
