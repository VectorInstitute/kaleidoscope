FROM marshw/pytriton_base_image:0.1.4

ARG DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /build
WORKDIR /build

RUN apt-key del 7fa2af80 && \
    apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends curl && \
    curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends \
    git \
    python3-pip python3-dev

# Install Python packages
RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install -r requirements.txt
RUN pip install -e .
