FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Install python 3.8
RUN apt-get update \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gpg-agent \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common \
	&& add-apt-repository -y ppa:deadsnakes

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3.8 \
	&& apt-get clean

# Install and upgrade pip
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-pip \
	&& pip install --upgrade pip

# Install PyTriton
RUN pip install -U https://github.com/triton-inference-server/pytriton/releases/download/v0.1.4/pytriton-0.1.4-py3-none-manylinux_2_31_x86_64.whl

# Install other base dependencies
RUN python3 -m pip install datasets==2.10.1 pandas==1.5.3 requests==2.28.2 transformers==4.29.2