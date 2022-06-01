#! /bin/sh

if [ $# -eq 3 ]; then
    echo "Building image : ${3%:*}-user:${3#*:}"
    DOCKER_BUILDKIT=1 docker build \
        --file Dockerfile.Adduser \
        --build-arg UNAME_ARG=$1 \
        --build-arg UID_ARG=$2 \
        --build-arg BASE_IMAGE=$3 \
        --tag "${3%:*}-user:${3#*:}" \
        .
    echo "Built image : ${3%:*}-user:${3#*:}\n"

else

    echo "3 arguments are expected : uname, uid, {BASE_IMAGE}"
    exit 1

fi
