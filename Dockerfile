FROM horovod/horovod:0.18.1-tf1.14.0-torch1.2.0-mxnet1.5.0-py3.6
WORKDIR usr/src/dqn
COPY . .
RUN pip install --upgrade protobuf -i https://pypi.mirrors.ustc.edu.cn/simple/
RUN pip install atari-py -i https://pypi.mirrors.ustc.edu.cn/simple/
RUN pip install tensorboardX -i https://pypi.mirrors.ustc.edu.cn/simple/
RUN pip install zmq -i https://pypi.mirrors.ustc.edu.cn/simple/
RUN pip install gym -i https://pypi.mirrors.ustc.edu.cn/simple/