FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

# install basic tools
RUN apt-get update && \
    apt-get install vim wget tree htop -y && \
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

# download miniconda
RUN wget -q -P /tmp https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh
ENV PATH /opt/conda/bin:$PATH

# install env
RUN conda create -n dock_gpu python=3.9 \
                             tqdm=4.63.0 \
                             matplotlib=3.3.4 \
                             seaborn=0.11.2 \
                             numpy=1.21.2 -y

# import path & clean up
ENV PATH /opt/conda/envs/dock_gpu/bin:$PATH
RUN echo "source activate dock_gpu" >> ~/.bashrc
RUN rm -rf /opt/conda/pkgs/*

# download test data
RUN mkdir /DB && \
    cd /DB && \
    wget https://dp-tech-zhangjiakou.oss-cn-zhangjiakou.aliyuncs.com/DPLC/LITPCBA/forDockGPU-LITPCBA.tar.bz2 && \
    tar -xvf forDockGPU-LITPCBA.tar.bz2 && \
    rm forDockGPU-LITPCBA.tar.bz2 && \
    wget https://dp-tech-zhangjiakou.oss-cn-zhangjiakou.aliyuncs.com/DPLC/CASF-2016/CASF-2016-fordock.tar.gz2 &&\
    tar -xvf CASF-2016-fordock.tar.gz2 && \
    rm CASF-2016-fordock.tar.gz2

