FROM ubuntu:22.04

RUN apt update
RUN apt install -y python3.10 python3-pip
RUN apt install -y vim git less

# Now install the required pip packages
WORKDIR /build
ADD requirements.txt .
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install nvidia-pytriton
