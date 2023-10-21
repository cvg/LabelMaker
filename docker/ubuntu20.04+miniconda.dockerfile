FROM ubuntu:20.04
WORKDIR /root
RUN apt-get update && apt-get -y install git curl wget make nano && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && chmod +x /root/Miniconda3-latest-Linux-x86_64.sh && /root/Miniconda3-latest-Linux-x86_64.sh -b && rm -rf /root/Miniconda3-latest-Linux-x86_64.sh && /root/miniconda3/bin/conda init bash
COPY ./.git /LabelMaker/.git
COPY ./3rdparty /LabelMaker/3rdparty
WORKDIR /LabelMaker
