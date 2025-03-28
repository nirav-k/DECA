FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y wget curl git build-essential

#Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.8.2-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.8.2-Linux-x86_64.sh

RUN conda install torchvision==0.8.2 torchaudio==0.7.2 -c pytorch && \
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath

WORKDIR /content/
COPY . /content/DECA/

RUN pip install -r DECA/requirements.txt