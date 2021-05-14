FROM nvcr.io/nvidia/pytorch:21.04-py3

RUN apt-get update && apt-get install -y curl gnupg  && rm -rf /var/lib/apt/lists/*

# 言語の設定
RUN apt-get update && apt-get install -y language-pack-ja-base language-pack-ja && rm -rf /var/lib/apt/lists/*
ENV LANG='ja_JP.UTF-8'

# 必要なもののインストール
RUN apt-get update && apt-get install -y p7zip-full zip && rm -rf /var/lib/apt/lists/*
RUN pip install natsort

# trtorchを展開
WORKDIR /root
RUN wget https://github.com/NVIDIA/TRTorch/releases/download/v0.3.0/libtrtorch-v0.3.0-cudnn8.1-tensorrt7.2-cuda11.1-libtorch-1.8.1.tar.gz.tar.gz
RUN tar xvf libtrtorch-v0.3.0-cudnn8.1-tensorrt7.2-cuda11.1-libtorch-1.8.1.tar.gz.tar.gz .

# 環境変数の設定
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/libtorch-1.8.1/lib:/root/trtorch/lib

# Miacisの導入
RUN git clone https://github.com/SakodaShintaro/Miacis
WORKDIR /root/Miacis
RUN git checkout libtorch-1.8.1
RUN sh /root/Miacis/scripts/download_libtorch.sh
WORKDIR /root/Miacis/src/cmake-build-release
RUN echo "git fetch" > update.sh && \
    echo "git reset --hard origin/libtorch-1.8.1" >> update.sh && \
    echo "cmake -DCMAKE_BUILD_TYPE=Release .." >> update.sh && \
    echo "make -j$(nproc) Miacis_shogi_categorical" >> update.sh && \
    chmod +x update.sh && \
    ./update.sh

# パラメータの取得
RUN wget https://github.com/SakodaShintaro/Miacis/releases/download/20210407/shogi_cat_bl10_ch256.model

# dotfileの導入
WORKDIR /root
RUN git clone https://github.com/SakodaShintaro/dotfiles && ./dotfiles/setup.sh
