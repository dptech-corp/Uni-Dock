# This docker image depends on nvidia-docker, see https://github.com/NVIDIA/nvidia-docker

FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

COPY ./src /opt/vina_gpu/src/
COPY ./build /opt/vina_gpu/build/
COPY ./example /opt/vina_gpu/example/
WORKDIR /opt/vina_gpu/

# ENV PYTHON_VERSION=3.8
RUN apt-get update 
RUN apt-get install -y wget vim

RUN mkdir packages; \
    cd packages; \
    wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz; \
    tar -xzvf boost_1_77_0.tar.gz; \
    rm boost_1_77_0.tar.gz; \
    cd boost_1_77_0/; \
    ./bootstrap.sh; \
    ./b2; \
    ./b2 install --prefix=/usr/local;

ENV LD_LIBRARY_PATH /usr/local/lib/:$LD_LIBRARY_PATH

ENV PATH /opt/vina_gpu/build/linux/release:$PATH

RUN cd /opt/vina_gpu/build/linux/release; make clean; make -j 4

ENTRYPOINT /bin/bash