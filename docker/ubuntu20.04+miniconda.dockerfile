FROM ubuntu:20.04
WORKDIR /
ENV TZ=Europe/Zurich
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone
RUN apt-get update && \
    apt-get -y install git curl wget make nano ffmpeg libsm6 libxext6 unzip && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /Miniconda3-latest-Linux-x86_64.sh && \
    /Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3 && \
    rm -rf /Miniconda3-latest-Linux-x86_64.sh && \
    /miniconda3/bin/conda init bash && \
    chmod -R 777 /miniconda3
RUN export PATH="/miniconda3/bin:$PATH" && conda config --set auto_activate_base false
COPY ./.git /LabelMaker/.git
COPY ./.gitmodules /LabelMaker/.gitmodules
COPY ./3rdparty /LabelMaker/3rdparty
COPY ./env /LabelMaker/env
COPY ./labelmaker /LabelMaker/labelmaker
COPY ./setup.py /LabelMaker/setup.py
WORKDIR /LabelMaker
ENV ENV_FOLDER /LabelMaker/env
SHELL ["/bin/bash", "-c"] 
RUN export PATH="/miniconda3/bin:$PATH" && \
    bash ${ENV_FOLDER}/00_initialize_labelmaker_local.sh 3.9 11.3 1.12.0 9.5.0 && \
    rm -rf /root/.cache/*
RUN export PATH="/miniconda3/bin:$PATH" && \
    source ${ENV_FOLDER}/activate_labelmaker_local.sh && \
    bash ${ENV_FOLDER}/01_pip_packages_install.sh && \
    rm -rf /root/.cache/*
RUN export PATH="/miniconda3/bin:$PATH" && \
    source ${ENV_FOLDER}/activate_labelmaker_local.sh && \
    bash ${ENV_FOLDER}/02.0_mask3d_detectron_2.sh && \
    rm -rf /root/.cache/*
RUN export PATH="/miniconda3/bin:$PATH" && \
    source ${ENV_FOLDER}/activate_labelmaker_local.sh && \
    bash ${ENV_FOLDER}/02.1_mask3d_minkowskiengine.sh && \
    rm -rf /root/.cache/*
RUN export PATH="/miniconda3/bin:$PATH" && \
    source ${ENV_FOLDER}/activate_labelmaker_local.sh && \
    bash ${ENV_FOLDER}/02.2_mask3d_others.sh && \
    rm -rf /root/.cache/*
RUN export PATH="/miniconda3/bin:$PATH" && \
    source ${ENV_FOLDER}/activate_labelmaker_local.sh && \
    bash ${ENV_FOLDER}/03_omnidata_hha_cmx.sh && \
    rm -rf /root/.cache/*
RUN export PATH="/miniconda3/bin:$PATH" && \
    source ${ENV_FOLDER}/activate_labelmaker_local.sh && \
    bash ${ENV_FOLDER}/04_grounded_sam.sh && \
    rm -rf /root/.cache/*
RUN export PATH="/miniconda3/bin:$PATH" && \
    source ${ENV_FOLDER}/activate_labelmaker_local.sh && \
    bash ${ENV_FOLDER}/05_ovseg.sh && \
    rm -rf /root/.cache/*
RUN export PATH="/miniconda3/bin:$PATH" && \
    source ${ENV_FOLDER}/activate_labelmaker_local.sh && \
    bash ${ENV_FOLDER}/06_internimage.sh && \
    rm -rf /root/.cache/*
RUN export PATH="/miniconda3/bin:$PATH" && \
    source ${ENV_FOLDER}/activate_labelmaker_local.sh && \
    bash ${ENV_FOLDER}/07_install_labelmaker.sh && \
    rm -rf /root/.cache/*
RUN chmod -R 777 /miniconda3/envs/labelmaker
RUN export PATH="/miniconda3/bin:$PATH" && \
    bash ${ENV_FOLDER}/10_initialize_sdfstudio_local.sh 3.10 11.3 && \
    rm -rf /root/.cache/*
RUN export PATH="/miniconda3/bin:$PATH" && \
    source ${ENV_FOLDER}/activate_sdfstudio_local.sh && \
    bash ${ENV_FOLDER}/11_pip_packages_install.sh && \
    rm -rf /root/.cache/*
RUN export PATH="/miniconda3/bin:$PATH" && \
    source ${ENV_FOLDER}/activate_sdfstudio_local.sh && \
    bash ${ENV_FOLDER}/12_install_tcnn.sh && \
    rm -rf /root/.cache/*
RUN chmod -R 777 /miniconda3/envs/sdfstudio
