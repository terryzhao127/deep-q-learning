FROM continuumio/anaconda3 
RUN apt-get update
RUN apt-get install vim -y
RUN apt-get install tmux -y
RUN apt-get install lrzsz -y
RUN apt-get install autoconf -y
RUN apt-get install automake -y
RUN apt-get install libtool -y
COPY env.yml /

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
RUN conda config --set show_channel_urls yes	
RUN conda env create -f env.yml
RUN conda install pyzmq -y
RUN conda install protobuf -y

COPY dqn.py /
COPY actor.py /
COPY learner.py /
#COPY cartpole-dqn.h5 /save/

git clone https://github.com/google/protobuf.git  
cd protobuf/ 
./autogen.sh 

COPY data.proto /
RUN protoc -I=. --python_out=.  data.proto