# Running this Docker image requires Docker to support NVIDIA GPUs. Please make sure NVIDIA Container Toolkit is configured.
# See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html https://github.com/NVIDIA/nvidia-container-toolkit

ARG CUDA_VERSION=12.0.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

RUN apt-get update && apt install -y cmake \
    libboost-system-dev libboost-thread-dev libboost-serialization-dev libboost-filesystem-dev libboost-program-options-dev libboost-timer-dev git

COPY . /opt/unidock

RUN cd /opt/unidock && \
    cmake -B build && \
    cmake --build build -j`nprocs` && \
    cmake --install build && \
    rm -r /opt/unidock

# Build this Docker image:
# cd Uni-Dock-Dev/unidock && \
# docker build . -f Dockerfile -t unidock --build-arg CUDA_VERSION=12.2.0
