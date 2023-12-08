FROM ubuntu:16.04
WORKDIR /root
ENV TZ=Europe/Zurich
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone
RUN apt-get update && apt-get -y install git curl wget make nano libgl1 libglib2.0-0 ffmpeg libsm6 libxext6 && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && chmod +x /root/Miniconda3-latest-Linux-x86_64.sh && /root/Miniconda3-latest-Linux-x86_64.sh -b && rm -rf /root/Miniconda3-latest-Linux-x86_64.sh && /root/miniconda3/bin/conda init bash
RUN export PATH="/root/miniconda3/bin:$PATH" && conda config --set auto_activate_base false
COPY ./.git /LabelMaker/.git
COPY ./.gitmodules /LabelMaker/.gitmodules
COPY ./3rdparty /LabelMaker/3rdparty
COPY ./env_v2 /LabelMaker/env_v2
COPY ./labelmaker /LabelMaker/labelmaker
COPY ./setup.py /LabelMaker/setup.py
WORKDIR /LabelMaker
RUN export PATH="/root/miniconda3/bin:$PATH" && bash env_v2/install_labelmaker_env.sh 3.9 11.8 2.0.0 10.4.0 && rm -rf /root/.cache/*
RUN export PATH="/root/miniconda3/bin:$PATH" && conda activate labelmaker && export conda_home=/root/miniconda3/envs/labelmaker && export CUDA_HOST_COMPILER="$conda_home/bin/gcc" && export CUDA_PATH="$conda_home" && export CUDA_HOME=$CUDA_PATH && pip install /LabelMaker/3rdparty/Grounded-Segment-Anything/GroundingDINO/ && rm -rf /root/.cache/*
RUN export PATH="/root/miniconda3/bin:$PATH" && bash env_v2/install_sdfstudio_env.sh 3.10 11.3 && rm -rf /root/.cache/*
