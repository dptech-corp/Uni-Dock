FROM dp-harbor-registry.cn-zhangjiakou.cr.aliyuncs.com/dplc/vina_gpu:base_env-v0514

ENV PATH /opt/vina_gpu/build/linux/release:$PATH

# install
COPY ./src /opt/vina_gpu/src/
COPY ./build /opt/vina_gpu/build/
COPY ./example /opt/vina_gpu/example/
RUN /bin/bash -c "cd /opt/vina_gpu/build/linux/release; make clean; make -j 4"
ENV LD_LIBRARY_PATH /opt/lib/packages/boost1_77/lib/:$LD_LIBRARY_PATH