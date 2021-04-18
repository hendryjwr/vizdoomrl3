FROM nvidia/cuda:10.2-base-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive 


RUN apt-get update && apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev apt-utils liblzma-dev

RUN wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz && tar xzvf Python-3.7.9.tgz && cd Python-3.7.9 && ./configure && make && make install 

RUN apt-get install -y git


RUN apt-get install -y build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip cmake

RUN apt-get install -y libboost-all-dev

RUN apt-get install -y liblua5.1-dev

COPY . /app

WORKDIR /app


RUN git clone https://github.com/shakenes/vizdoomgym.git

RUN cd vizdoomgym && pip3 install -e .

RUN pip3 install -r requirements.txt

