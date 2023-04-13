FROM --platform=linux/amd64 ubuntu:focal-20230308


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

# Install other requirements
COPY requirements.txt ./
RUN pip install -r requirements.txt


#COPY test_triton_install.py .

#CMD ["cat", "/etc/os-release"]
#ENTRYPOINT ["python3"]

#CMD ["ldd", "--version"]
#CMD ["test_triton_install.py"]
