# Most of this Dockerfile is based on NVIDIA's (Apache-licensed) Dockerfile provided in the fastertransformer_backend repo:
# https://github.com/triton-inference-server/fastertransformer_backend/blob/main/docker/Dockerfile

ARG TRITON_VERSION=22.01
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3

FROM ${BASE_IMAGE} as server-builder

RUN   apt-key del 7fa2af80
ADD   https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb .
RUN   dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    zip unzip wget build-essential autoconf autogen gdb git \
    python3.8 python3-pip python3-dev rapidjson-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone https://github.com/triton-inference-server/fastertransformer_backend.git
RUN cd fastertransformer_backend && git checkout -b t5_gptj_blog remotes/origin/dev/t5_gptj_blog

WORKDIR /workspace/build/

# CMake
RUN CMAKE_VERSION=3.18 && \
    CMAKE_BUILD=3.18.4 && \
    wget -nv https://cmake.org/files/v${CMAKE_VERSION}/cmake-${CMAKE_BUILD}.tar.gz && \
    tar -xf cmake-${CMAKE_BUILD}.tar.gz && \
    cd cmake-${CMAKE_BUILD} && \
    ./bootstrap --parallel=$(grep -c ^processor /proc/cpuinfo) -- -DCMAKE_USE_OPENSSL=OFF && \
    make -j"$(grep -c ^processor /proc/cpuinfo)" install && \
    cd /workspace/build/ && \
    rm -rf /workspace/build/cmake-${CMAKE_BUILD}

# backend build
WORKDIR /workspace/build/fastertransformer_backend
RUN cp -r /workspace/fastertransformer_backend/cmake .
RUN cp -r /workspace/fastertransformer_backend/src .
RUN cp -r /workspace/fastertransformer_backend/CMakeLists.txt .
RUN cp -r /workspace/fastertransformer_backend/README.md .
RUN cp -r /workspace/fastertransformer_backend/LICENSE .

ARG FORCE_BACKEND_REBUILD=0
RUN mkdir build -p && \
    cd build && \
    cmake \
      -D CMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/opt/tritonserver \
      -D TRITON_COMMON_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_CORE_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_BACKEND_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      .. && \
    make -j"$(grep -c ^processor /proc/cpuinfo)" install

# Add a new layer to the image for the actual server
# This will clean up all the files left behind from the instructions above
FROM ${BASE_IMAGE} as server

ENV NCCL_LAUNCH_MODE=GROUP

COPY --from=server-builder /opt/tritonserver/backends/fastertransformer /opt/tritonserver/backends/fastertransformer

# set workspace
ENV WORKSPACE /workspace
WORKDIR /workspace

RUN   apt-key del 7fa2af80
ADD   https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb .
RUN   dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# Need to install OpenMPI before installing PyTorch
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends openmpi-bin openmpi-doc libopenmpi-dev

# Remove previous openmpi version
RUN rm -rf /opt/hpcx/ompi
RUN rm -rf /usr/local/mpi

# Install openmpi from source
RUN wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.6.tar.gz && \
    gunzip -c openmpi-3.0.6.tar.gz | tar xf - && \
    cd openmpi-3.0.6 && \
    ./configure --prefix=/usr/local/mpi --with-cuda && \
    make all install && \
    export PATH=$PATH:/usr/local/mpi/bin && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/mpi/lib/

# Install pytorch
# RUN pip3 install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html  && \
#     pip3 install --extra-index-url https://pypi.ngc.nvidia.com regex fire ipywidgets tritonclient[all]
RUN pip3 install torch -f https://download.pytorch.org/whl/torch_stable.html  && \
    pip3 install --extra-index-url https://pypi.ngc.nvidia.com regex fire ipywidgets tritonclient[all]

## for debug
#RUN apt-get update -q && \
#    apt-get install -y --no-install-recommends openssh-server zsh tmux mosh locales-all clangd sudo cmake xz-utils zstd libz-dev git-lfs&& \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

# RUN sed -i 's/#X11UseLocalhost yes/X11UseLocalhost no/g' /etc/ssh/sshd_config
# RUN mkdir /var/run/sshd -p

# temporal changes only for the blogpost about GPT-J and T5
RUN ln -sf /usr/bin/python3.8 /usr/bin/python

WORKDIR /workspace
RUN git clone https://github.com/triton-inference-server/fastertransformer_backend.git

# Upgrade pip3
RUN pip install --upgrade pip

# Install PyTriton
RUN pip install -U https://github.com/triton-inference-server/pytriton/releases/download/v0.1.4/pytriton-0.1.4-py3-none-manylinux_2_31_x86_64.whl

# Install other requirements
ADD requirements.txt .
RUN pip install -r requirements.txt

# Now clone and build FasterTransformer
# NOTE: This will take 1-2 hours and your computer will be completely useless while compiling. Save all your work first!
#RUN git clone https://github.com/NVIDIA/FasterTransformer.git
#RUN mkdir -p FasterTransformer/build
#WORKDIR /workspace/FasterTransformer/build
#RUN git submodule init && git submodule update
#RUN cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
#RUN make -j4

