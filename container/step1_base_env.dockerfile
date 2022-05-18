FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

# install basic tools
RUN apt-get update && \
    apt-get install vim wget -y && \
    rm -rf /var/lib/apt/lists/*

# install boost
RUN mkdir /opt/packages && \
    cd /opt/packages && \
    wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz && \
    tar -xzvf boost_1_77_0.tar.gz && \
    rm boost_1_77_0.tar.gz && \
    cd boost_1_77_0/ && \
    ./bootstrap.sh && \
    ./b2 && \
    ./b2 install --prefix=/opt/lib/packages/boost1_77
ENV LD_LIBRARY_PATH /opt/lib/packages/boost1_77/lib/:$LD_LIBRARY_PATH