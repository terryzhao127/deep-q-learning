FROM horovod/horovod:0.18.1-tf1.14.0-torch1.2.0-mxnet1.5.0-py3.6

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pip protobuf
RUN pip install zmq gym tensorboardX
RUN pip install gym[atari]

RUN apt-get update && apt-get install libgl1-mesa-glx libglib2.0-dev ffmpeg -y
WORKDIR /workplace
CMD ["/bin/bash"]
