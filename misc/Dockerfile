FROM nvcr.io/nvidia/pytorch:25.01-py3 

RUN apt update --allow-insecure-repositories
RUN apt install -y tmux
COPY misc/tmux.conf /etc/

ENV PYTHONPATH "/data/:/data/vae_nn_v2_public/"
ENV TF_CPP_MIN_LOG_LEVEL 2

COPY misc/requirements.txt /data/
RUN pip install -r /data/requirements.txt

RUN echo "root:x" | chpasswd
ARG currUID
RUN useradd -m -u $currUID ka
USER ka

WORKDIR /data/