FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# install dependencies
RUN printf 'deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse\n \
    deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse\n \
    deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse\n \
    deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse\n' > /etc/apt/sources.list \
    && apt-get clean && apt-get update \
    && apt install -y cmake build-essential zip unzip vim git wget \
    libboost-system-dev libboost-thread-dev libboost-serialization-dev libboost-filesystem-dev libboost-program-options-dev libboost-timer-dev \
    openjdk-19-jdk mesa-common-dev libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# install unidock
WORKDIR /tmp
RUN git clone https://github.com/dptech-corp/Uni-Dock.git
RUN cd Uni-Dock/unidock \
    && cmake -B build \
    && cmake --build build -j`nprocs` \
    && cmake --install build \
    && rm -r /tmp/Uni-Dock

# install conda
RUN wget -q -O /tmp/conda.sh \
    https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/Miniforge3%2023.3.1-1/Miniforge3-23.3.1-1-Linux-x86_64.sh \
    && bash /tmp/conda.sh -b -p /opt/conda \
    && rm /tmp/conda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda config --set show_channel_urls yes && \
    printf "\
    channels: \n\
    - defaults \n\
    show_channel_urls: true \n\
    default_channels: \n\
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \n\
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r \n\
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2 \n\
    custom_channels: \n\
    conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
    msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
    bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
    menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
    pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
    pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
    simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
    " > ~/.condarc && conda clean -i && pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN mamba install -y cmake boost -c conda-forge

# install python packages
WORKDIR /opt
RUN mamba install -y requests ipython \
    numpy pandas scipy matplotlib scikit-learn networkx \
    rdkit openbabel spyrmsd -c conda-forge
RUN pip install pydantic pydantic_settings lmdb multiprocess
RUN rm -rf /opt/conda/pkgs/*

# install CDPKit
WORKDIR /tmp
RUN git clone https://github.com/molinfo-vienna/CDPKit.git
RUN cd CDPKit && mkdir build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/opt/CDPKit \
    && make -j4 && make install
RUN rm -rf /tmp/CDPKit
ENV PATH=/opt/CDPKit/Bin:$PATH

