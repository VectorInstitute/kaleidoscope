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

# Install the Stable Diffusion libraries
RUN git clone https://github.com/CompVis/stable-diffusion
WORKDIR /build/stable-diffusion
RUN pip install -e .

# Install Python packages
RUN pip install \
    torch==2.0.0 \
    torchvision \
    omegaconf==2.3.0 \
    einops==0.7.0 \
    pytorch-lightning==1.7.7 \
    torchmetrics==0.11.4 \
    taming-transformers-rom1504==0.0.6 \
    clip==0.2.0 \
    kornia==0.6

WORKDIR /build