FROM ubuntu:20.04
WORKDIR /
ENV TZ=Europe/Zurich
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone
RUN apt-get update && \
    apt-get -y install git curl wget make nano ffmpeg libsm6 libxext6 unzip && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /root/Miniconda3-latest-Linux-x86_64.sh && \
    /root/Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3 && \
    rm -rf /root/Miniconda3-latest-Linux-x86_64.sh && \
    /miniconda3/bin/conda init bash && \
    chmod -R 777 /miniconda3
RUN export PATH="/miniconda3/bin:$PATH" && conda config --set auto_activate_base false
COPY ./.git /LabelMaker/.git
COPY ./.gitmodules /LabelMaker/.gitmodules
COPY ./3rdparty /LabelMaker/3rdparty
COPY ./env_v2 /LabelMaker/env_v2
COPY ./labelmaker /LabelMaker/labelmaker
COPY ./scripts /LabelMaker/scripts
COPY ./setup.py /LabelMaker/setup.py
WORKDIR /LabelMaker
RUN export PATH="/miniconda3/bin:$PATH" && \
    bash env_v2/install_labelmaker_env.sh 3.9 11.3 1.12.0 9.5.0 && \
    rm -rf /root/.cache/* && \
    chmod -R 777 /miniconda3/envs/labelmaker
RUN export PATH="/miniconda3/bin:$PATH" && \
    bash env_v2/install_sdfstudio_env.sh 3.10 11.3 && \
    rm -rf /root/.cache/* && \
    chmod -R 777 /miniconda3/envs/sdfstudio
