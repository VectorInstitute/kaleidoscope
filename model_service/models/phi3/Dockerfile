FROM ubuntu:22.04

RUN apt update
RUN apt install -y python3.10 python3-pip

# Now install the required pip packages
RUN python3 -m pip install nvidia-pytriton torch transformers protobuf accelerate
