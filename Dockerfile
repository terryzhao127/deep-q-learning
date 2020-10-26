FROM python:3.7
RUN apt-get update
RUN apt-get install vim -y
RUN apt-get install tmux -y
RUN apt-get install lrzsz -y
RUN apt-get install autoconf -y
RUN apt-get install automake -y
RUN apt-get install libtool -y

RUN pip install tensorflow==1.15
RUN pip install pyzmq
RUN pip install gym gym[atari]
#RUN pip install protobuf

COPY actor.py /
COPY learner.py /

COPY data.proto /
RUN git clone https://github.com/google/protobuf.git  
WORKDIR protobuf
RUN git submodule update --init --recursive
RUN ./autogen.sh
RUN ./configure --prefix=$INSTALL_DIR
RUN make -j4
RUN make check
RUN make install
RUN ldconfig
WORKDIR /
RUN protoc -I=. --python_out=.  data.proto
RUN rm -r protobuf
