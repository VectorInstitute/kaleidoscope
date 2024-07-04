FROM ubuntu:22.04

# Update apt packages, notably we need g++20
RUN apt update
RUN apt install -y wget vim less
RUN apt install -y build-essential pkg-config
RUN apt install -y zlib1g zlib1g-dev 
RUN apt install -y gcc-10 gcc-10-base gcc-10-doc g++-10
RUN apt install -y libstdc++-10-dev libstdc++-10-doc
RUN apt install -y libssl-dev

# Install python 3.11.9. Is there a better way than installing from source?
WORKDIR /build
RUN wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
RUN tar zxvf Python-3.11.9.tgz
WORKDIR /build/Python-3.11.9
RUN ./configure --enable-optimizations
RUN make install

# Install moonshot library
RUN python3 -m pip install "aiverify-moonshot[all]"



