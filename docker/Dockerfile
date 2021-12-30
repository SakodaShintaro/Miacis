FROM nvcr.io/nvidia/pytorch:21.04-py3

# 言語の設定
RUN apt-get update && apt-get install -y language-pack-ja-base language-pack-ja && rm -rf /var/lib/apt/lists/*
ENV LANG='ja_JP.UTF-8'

# 必要なもののインストール
RUN apt-get update && apt-get install -y p7zip-full zip && rm -rf /var/lib/apt/lists/*
RUN pip install natsort

# WORKDIRを設定
WORKDIR /root

# TensorRTをリンクするため、所定の位置にシンボリックリンクを張る
# パスが通っているはずなので本当はいらない気がするんだけど、ないとリンクエラーになる(include側は本当に不要かも)
# リンクする順番の問題とかだったりするかもしれない
RUN mkdir TensorRT-7.2.2.3
RUN ln -s /usr/include/x86_64-linux-gnu/ ./TensorRT-7.2.2.3/include
RUN mkdir TensorRT-7.2.2.3/lib
RUN ln -s /usr/lib/x86_64-linux-gnu/libnv*.so ./TensorRT-7.2.2.3/lib/

# 環境変数の設定
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/libtorch-1.10.1/lib

# Miacisの導入
RUN git clone https://github.com/SakodaShintaro/Miacis
RUN sh /root/Miacis/scripts/download_libtorch.sh
WORKDIR /root/Miacis/build
RUN echo "git fetch" > update.sh && \
    echo "git reset --hard origin/master" >> update.sh && \
    echo "cmake -DCMAKE_BUILD_TYPE=Release ../src" >> update.sh && \
    echo "make -j$(nproc) Miacis_shogi_categorical" >> update.sh && \
    chmod +x update.sh && \
    ./update.sh

# dotfileの導入
WORKDIR /root
RUN git clone https://github.com/SakodaShintaro/dotfiles && ./dotfiles/setup.sh
