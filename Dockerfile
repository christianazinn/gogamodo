FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y wget build-essential g++ vim && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH /opt/conda/bin:$PATH

WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get install -y fluidsynth timidity git tmux

RUN conda env create -f environment.yml && \
    conda init bash && \
    echo "conda activate gigamidicaps" >> ~/.bashrc

RUN conda run -n gigamidicaps pip install -r requirements.txt

RUN conda run -n gigamidicaps git clone https://github.com/christianazinn/chord-extractor.git && \
    cd chord-extractor && \
    conda run -n gigamidicaps pip install -e . && \
    cd ..