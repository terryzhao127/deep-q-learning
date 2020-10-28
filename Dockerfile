FROM horovod/horovod:0.18.1-tf1.14.0-torch1.2.0-mxnet1.5.0-py3.6

RUN pip install zmq gym \
    && pip install --upgrade protobuf

WORKDIR /workplace
CMD ["/bin/bash"]