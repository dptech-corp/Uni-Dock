ARG BASE_IMAGE=dptechnology/unidock:latest

FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get clean && apt-get update \
    && apt-get install -y build-essential zip unzip vim git wget

RUN wget --quiet -O conda.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh \
    && bash conda.sh -b -p /opt/conda \
    && rm conda.sh
ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet -O CDPKit.sh https://github.com/molinfo-vienna/CDPKit/releases/download/v1.1.0/CDPKit-1.1.0-Linux-x86_64.sh && \
    echo y | bash CDPKit.sh --cpack_skip_license --include-subdir && \
    rm CDPKit.sh

WORKDIR /opt
RUN mamba install -y ipython requests tqdm lmdb openbabel ambertools openmm -c conda-forge

COPY . /opt/unidock_tools
RUN cd /opt/unidock_tools && \
    pip install . && \
    rm -r /opt/unidock_tools
