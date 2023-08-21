# Running this Docker image requires Docker to support NVIDIA GPUs. Please make sure NVIDIA Container Toolkit is configured.
# See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html https://github.com/NVIDIA/nvidia-docker

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

COPY . /opt/unidock

RUN apt-get update && apt install -y cmake \
    libboost-system-dev libboost-thread-dev libboost-serialization-dev libboost-filesystem-dev libboost-program-options-dev

RUN cd /opt/unidock && \
    cmake -B build && \
    cmake --build build -j`nprocs` && \
    cmake --install build && \
    rm -r /opt/unidock
