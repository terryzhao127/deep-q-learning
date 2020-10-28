FROM horovod/horovod:0.18.1-tf1.14.0-torch1.2.0-mxnet1.5.0-py3.6
WORKDIR usr/src/dqn
COPY . .
RUN pip install --upgrade protobuf
RUN pip install zmq
RUN pip install gym