FROM nvcr.io/nvidia/pytorch:21.08-py3

# 言語の設定
RUN apt-get update && apt-get install -y language-pack-ja-base language-pack-ja && rm -rf /var/lib/apt/lists/*
ENV LANG='ja_JP.UTF-8'

# 必要なもののインストール
RUN apt-get update && apt-get install -y p7zip-full zip && rm -rf /var/lib/apt/lists/*
RUN pip install natsort

# trtorchを展開
WORKDIR /root
RUN wget https://github.com/NVIDIA/TRTorch/releases/download/v0.4.0/libtrtorch-v0.4.0-cudnn8.2-tensorrt8.0-cuda11.1-libtorch-1.9.0.tar.gz
RUN tar xvf libtrtorch-v0.4.0-cudnn8.2-tensorrt8.0-cuda11.1-libtorch-1.9.0.tar.gz .

# TensorRTをリンクするため、所定の位置にシンボリックリンクを張る
# パスが通っているはずなので本当はいらない気がするんだけど、ないとリンクエラーになる(include側は本当に不要かも)
# リンクする順番の問題とかだったりするかもしれない
RUN mkdir TensorRT-8.0.1.6
RUN ln -s /usr/include/x86_64-linux-gnu/ ./TensorRT-8.0.1.6/include
RUN mkdir TensorRT-8.0.1.6/lib
RUN ln -s /usr/lib/x86_64-linux-gnu/libnv*.so ./TensorRT-8.0.1.6/lib/

# 環境変数の設定
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/libtorch-1.9.0-cu111/lib

# Miacisの導入
RUN git clone https://github.com/SakodaShintaro/Miacis -b update-trtorch
RUN sh /root/Miacis/scripts/download_libtorch.sh
WORKDIR /root/Miacis/src/cmake-build-release
RUN echo "cmake -DCMAKE_BUILD_TYPE=Release .." > update.sh && \
    echo "make -j$(nproc) Miacis_shogi_categorical" >> update.sh && \
    chmod +x update.sh && \
    ./update.sh

# パラメータ,キャリブレーションキャッシュの取得
# そのままでは動かないのでコメントアウト
# RUN wget https://github.com/SakodaShintaro/Miacis/releases/download/20210407/shogi_cat_bl10_ch256.model
# RUN wget https://github.com/SakodaShintaro/Miacis/releases/download/20210407/calibration_cache_file.txt

# dotfileの導入
WORKDIR /root
RUN git clone https://github.com/SakodaShintaro/dotfiles && ./dotfiles/setup.sh
