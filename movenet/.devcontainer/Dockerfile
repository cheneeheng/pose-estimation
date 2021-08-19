## For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

RUN apt-get update -y
RUN apt-get install -y sudo nano unzip wget curl git

ENV TZ=Europe/Berlin
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

# Opencv
RUN apt-get install -y python3-opencv

# Python 3.8
RUN apt-get install -y python3.8 python3.8-dev
RUN apt-get install -y python3-pip

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
RUN pip3 install --upgrade pip
COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

RUN apt-get install ffmpeg -y
RUN apt-get install net-tools -y

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 1000 --disabled-password --gecos "" chen
RUN echo "chen:chen" | chpasswd 
RUN usermod -aG sudo chen
USER chen
WORKDIR /home/chen

RUN git clone https://github.com/tensorflow/docs && pip install docs/.

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["python", "-m", "run"]

EXPOSE 8000-9000
EXPOSE 8010/udp
