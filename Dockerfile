FROM weicc/tf1.4-horovod-protobuf:2.0
MAINTAINER weicc<ccwei@mail.ustc.edu.cn>

WORKDIR /workspace

RUN pip3 install --upgrade pip
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install tensorboardX 
RUN pip3 install scikit-build
RUN pip3 install cmake 
RUN pip3 --default-timeout=100 install gym[atari]
# RUN pip3 --default-timeout=100 install opencv-python 

CMD "/bin/bash"
