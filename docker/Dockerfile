FROM nvcr.io/nvidia/pytorch:22.04-py3

# 言語の設定
RUN apt-get update && apt-get install -y language-pack-ja-base language-pack-ja
ENV LANG='ja_JP.UTF-8'

# 必要なもののインストール
RUN apt-get install -y p7zip-full zip
RUN pip3 install natsort cshogi timm

# WORKDIRを設定
WORKDIR /root

# TensorRTをリンクするため、所定の位置にシンボリックリンクを張る
# パスが通っているはずなので本当はいらない気がするんだけど、ないとリンクエラーになる(include側は本当に不要かも)
# リンクする順番の問題とかだったりするかもしれない
ARG trt_dir_name="TensorRT-8.2.2.1"
RUN mkdir ${trt_dir_name}
RUN ln -s /usr/include/x86_64-linux-gnu/ ./${trt_dir_name}/include
RUN mkdir ${trt_dir_name}/lib
RUN ln -s /usr/lib/x86_64-linux-gnu/libnv*.so ./${trt_dir_name}/lib/

# Miacisの導入
RUN git clone https://github.com/SakodaShintaro/Miacis -b master
WORKDIR /root/Miacis/build
RUN echo "cmake -DCMAKE_BUILD_TYPE=Release ../src" > update.sh && \
    echo "make -j$(nproc) Miacis_shogi_categorical" >> update.sh && \
    chmod +x update.sh && \
    ./update.sh

# dotfileの導入
WORKDIR /root
RUN git clone https://github.com/SakodaShintaro/dotfiles && ./dotfiles/setup.sh
