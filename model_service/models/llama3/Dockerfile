FROM ubuntu:22.04

RUN apt update
RUN apt install -y python3.10 python3-pip
RUN apt install -y vim git less

# Install the pytriton package
WORKDIR /build
RUN python3 -m pip install nvidia-pytriton

# Install the llama package
RUN git clone https://github.com/meta-llama/llama3
WORKDIR llama3
RUN pip install -r requirements.txt
RUN pip install -e .

# Install other dependencies
RUN pip3 install cloudpickle==2.2.1 einops==0.6.1
