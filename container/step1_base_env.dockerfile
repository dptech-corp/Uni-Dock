FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

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

RUN conda create -y -n mgltools mgltools=1.5.7 \ 
    autogrid=4.2.6 -c bioconda -c conda-forge
# import path & clean up
ENV PATH /opt/conda/envs/dock_gpu/bin:/opt/conda/envs/mgltools/bin:$PATH
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

