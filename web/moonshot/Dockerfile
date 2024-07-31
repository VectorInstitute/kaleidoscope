FROM ubuntu:22.04

# Update + install apt packages
RUN apt update
RUN apt-get update
RUN apt install -y wget vim less git iproute2 iputils-ping sqlite3
RUN apt install -y build-essential pkg-config
RUN apt install -y zlib1g zlib1g-dev 
RUN apt install -y gcc-10 gcc-10-base gcc-10-doc g++-10
RUN apt install -y libstdc++-10-dev libstdc++-10-doc
RUN apt install -y libssl-dev libbz2-dev libsqlite3-dev libffi-dev

# Install python 3.11.9. Is there a better way than installing from source?
WORKDIR /build
RUN wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
RUN tar zxvf Python-3.11.9.tgz
WORKDIR /build/Python-3.11.9
RUN ./configure --enable-optimizations
RUN make install
RUN ln -s /usr/local/bin/pip3 /usr/local/bin/pip

# Install nodejs and npm from source (apt install will use python 3.10 which we don't want)
WORKDIR /build
RUN wget https://nodejs.org/dist/v20.15.0/node-v20.15.0.tar.gz
RUN tar zxvf node-v20.15.0.tar.gz
WORKDIR /build/node-v20.15.0
RUN ./configure
RUN make -j 8 install

# Install moonshot requirements
WORKDIR /build
RUN git clone https://github.com/aiverify-foundation/moonshot
WORKDIR /build/moonshot
RUN python3 -m pip install -r requirements.txt

# Install additional python libraries not included in requirements
# TODO: These packages should really be pinned, but the dependencies are insane
RUN python3 -m pip install requests jupyter notebook kscope prisma bcrypt

# Install moonshot module
RUN python3 -m pip install "aiverify-moonshot[all]"
# The following command returns exit code 1 even though it works properly. Append "|| true" to force a 0 exit.
RUN python3 -m moonshot -i moonshot-data -i moonshot-ui || true

# Copy some custom code into the relevant locations
WORKDIR /build/moonshot
ADD ./kscope-connector.py /build/moonshot/moonshot-data/connectors
ADD ./kscope-llama3-8b.json /build/moonshot/moonshot-data/connectors-endpoints
ADD ./kscope-phi3-medium-128k-instruct.json /build/moonshot/moonshot-data/connectors-endpoints

# Add and install custom authentication code
WORKDIR /build/moonshot/moonshot-ui
RUN git remote add next-auth https://github.com/amrit110/moonshot-ui.git
RUN git pull next-auth fix_routing
RUN npm install
RUN prisma migrate dev --name init
RUN prisma generate
RUN npm run build
ADD ./package.json .
RUN python3 ./add_user.py

# Do some repackaging
WORKDIR /build/moonshot/moonshot-data/connectors-endpoints
RUN rm -rf azure*.json claude*.json huggingface*.json llm-judge*.json ollama*.json openai-gpt35*.json together*.json
WORKDIR /build/moonshot/moonshot-data/attack-modules
RUN rm -rf *.py

# Now start the moonshot interface
WORKDIR /build/moonshot

