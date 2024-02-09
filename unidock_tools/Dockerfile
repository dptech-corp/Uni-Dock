FROM registry.dp.tech/dp/vina_pipeline:unidock_v1.1_240130_hbond

RUN apt-get install wget vim zip unzip git -y
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
ENV PATH /opt/conda/bin:$PATH

# RUN conda config --set show_channel_urls yes && \
# echo -e "\
# channels: \n\
#   - defaults \n\
# show_channel_urls: true \n\
# default_channels: \n\
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \n\
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r \n\
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2 \n\
# custom_channels: \n\
#   conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
#   msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
#   bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
#   menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
#   pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
#   pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
#   simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud \n\
# " > ~/.condarc && conda clean -i && pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

RUN conda update -y -c conda-forge conda && conda install -y mamba -c conda-forge
RUN mamba install -y ipython -c conda-forge
