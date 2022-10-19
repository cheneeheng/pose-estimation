FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ENV LANG en_US.UTF-8

ARG UNAME_ARG
ARG UID_ARG

# ------------------------------------------------------------------------------
# Main apts 
RUN apt-get update -y
RUN apt-get install -y sudo nano unzip wget curl git

ENV TZ=Europe/Berlin
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

# ------------------------------------------------------------------------------
# Opencv
RUN apt-get install -y python3-opencv

# ------------------------------------------------------------------------------
# Python 3.8
RUN apt-get install -y python3.8 python3.8-dev
RUN apt-get install -y python3-pip
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# ------------------------------------------------------------------------------
# Install pip requirements
RUN pip3 install --upgrade pip
COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

RUN apt-get install ffmpeg -y
RUN apt-get install net-tools -y

# ------------------------------------------------------------------------------
# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u ${UID_ARG} --disabled-password --gecos "" ${UNAME_ARG}
RUN echo "${UNAME_ARG}:${UNAME_ARG}" | chpasswd 
RUN usermod -aG sudo ${UNAME_ARG}
USER ${UNAME_ARG}
WORKDIR /code/pose-estimation/movenet

RUN git clone https://github.com/tensorflow/docs && pip install docs/.

