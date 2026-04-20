ARG BASE_IMAGE=dptechnology/unidock:latest

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get clean && apt-get update \
    && apt-get install -y build-essential zip unzip vim git wget


RUN wget --quiet -O CDPKit.sh https://github.com/molinfo-vienna/CDPKit/releases/download/v1.1.0/CDPKit-1.1.0-Linux-x86_64.sh && \
    echo y | bash CDPKit.sh --cpack_skip_license --include-subdir && \
    rm CDPKit.sh

RUN wget --quiet -O conda.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh \
    && bash conda.sh -b -p /opt/conda \
    && rm conda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN mamba create -y -n unidock python=3.12 ipython openbabel -c conda-forge

SHELL ["/bin/bash", "-c"]

RUN printf '#!/bin/bash\n\
set -e\n\
source /opt/conda/etc/profile.d/conda.sh\n\
source /opt/conda/etc/profile.d/mamba.sh\n\
mamba activate unidock\n\
exec "$@"\n' > /opt/entrypoint.sh && chmod +x /opt/entrypoint.sh

RUN printf '\
source /opt/conda/etc/profile.d/conda.sh\n\
source /opt/conda/etc/profile.d/mamba.sh\n\
mamba activate unidock\n' > /opt/activate_unidock.sh

ENV BASH_ENV="/opt/activate_unidock.sh"
ENV PATH="/opt/conda/envs/unidock/bin:/opt/conda/bin:$PATH"

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["bash"]

WORKDIR /opt

COPY . /opt/unidock_tools
RUN cd /opt/unidock_tools && \
    pip install . && \
    rm -r /opt/unidock_tools
