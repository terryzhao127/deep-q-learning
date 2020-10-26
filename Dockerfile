FROM ubuntu:18.04

RUN apt-get update
RUN apt install build-essential -y
RUN apt install libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev -y
RUN apt-get install zlib1g-dev -y
RUN apt-get install python3.7 -y
RUN apt install python3-distutils curl -y
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.7 get-pip.py -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install gym gym[atari]
ADD tensorflow-1.15.0-cp37-cp37m-manylinux2010_x86_64.whl /workplace/tensorflow-1.15.0-cp37-cp37m-manylinux2010_x86_64.whl
RUN pip install /workplace/tensorflow-1.15.0-cp37-cp37m-manylinux2010_x86_64.whl
RUN pip install zmq

WORKDIR /workplace
CMD ["/bin/bash"]
