FROM opensciencegrid/tensorflow-gpu:1.4
MAINTAINER weicc<ccwei@mail.ustc.edu.cn>

WORKDIR /workspace

RUN pip3 install pyzmq gym \
&& rm -rf /var/lib/apt/lists/* 

CMD "/bin/bash"
