FROM opensciencegrid/tensorflow-gpu:1.4
MAINTAINER weicc<ccwei@mail.ustc.edu.cn>

WORKDIR /workspace

RUN apt-get -y install git unzip build-essential autoconf libtool g++-4.8 \
    && rm -rf /var/lib/apt/lists/*

ADD protobuf.zip /workspace/protobuf.zip
COPY openmpi-4.0.0.tar.gz /workspace/openmpi-4.0.0.tar.gz

# Install Protocol Buffer
RUN unzip /workspace/protobuf.zip && \
    cd ./protobuf-master && \
    ./autogen.sh && \
    ./configure && \
    make && \
    make install && \
    ldconfig && \
    make clean && \
    cd .. && \
    rm -r protobuf-master

# Install OpenMI (necessary for Horovod)
RUN mkdir /tmp/openmpi && \
    mv /workspace/openmpi-4.0.0.tar.gz /tmp/openmpi/openmpi-4.0.0.tar.gz && \
    cd /tmp/openmpi && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install Horovod
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 \
    pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple horovod && \
    ldconfig

# Other packages
RUN pip3 install pyzmq gym protobuf==3.13.0 \
    && rm -rf /var/lib/apt/lists/*

# Otherwise, gpus can not be used
RUN apt -y remove nvidia-*

CMD "/bin/bash"
