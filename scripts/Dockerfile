FROM nvcr.io/nvidia/pytorch:20.10-py3

RUN apt-get update && apt-get install -y curl gnupg  && rm -rf /var/lib/apt/lists/*

RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

RUN apt-get update && apt-get install -y bazel-3.7.1 && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/bazel-3.7.1 /usr/bin/bazel

RUN pip install notebook

# trtorchの導入
WORKDIR /opt
RUN git clone https://github.com/NVIDIA/TRTorch trtorch
WORKDIR /opt/trtorch
RUN git checkout b228bf239aadd5f104af38ea64416bdda5f0aa57
RUN cp /opt/trtorch/docker/WORKSPACE.cu.docker /opt/trtorch/WORKSPACE

# Workaround for bazel expecting both static and shared versions, we only use shared libraries inside container
RUN cp /usr/lib/x86_64-linux-gnu/libnvinfer.so /usr/lib/x86_64-linux-gnu/libnvinfer_static.a

WORKDIR /opt/trtorch
RUN bazel build //:libtrtorch --compilation_mode opt

WORKDIR /opt/trtorch/py

RUN pip install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension

# Locale is not set by default
RUN apt-get update && apt-get install -y locales ninja-build && rm -rf /var/lib/apt/lists/* && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN python3 setup.py install --use-cxx11-abi

RUN conda init bash

ENV LD_LIBRARY_PATH /opt/conda/lib/python3.6/site-packages/torch/lib:$LD_LIBRARY_PATH


# ここから自分の設定
# 言語の設定
RUN apt-get update && apt-get install -y language-pack-ja-base language-pack-ja && rm -rf /var/lib/apt/lists/*
ENV LANG='ja_JP.UTF-8'

# 必要なもののインストール
RUN apt-get update && apt-get install -y p7zip-full zip && rm -rf /var/lib/apt/lists/*
RUN pip install natsort

# trtorchを適切な場所へ展開
WORKDIR /root
RUN tar xvf /opt/trtorch/bazel-bin/libtrtorch.tar.gz .

# Miacisの導入
RUN git clone https://github.com/SakodaShintaro/Miacis
RUN ./Miacis/scripts/download_libtorch.sh
WORKDIR /root/Miacis/src/cmake-build-release
RUN echo "git fetch" > update.sh && \
    echo "git reset --hard origin/master" >> update.sh && \
    echo "cmake -DCMAKE_BUILD_TYPE=Release .." >> update.sh && \
    echo "make -j$(nproc)" >> update.sh && \
    chmod +x update.sh && \
    ./update.sh

# dotfileの導入
WORKDIR /root
RUN git clone https://github.com/SakodaShintaro/dotfiles && ./dotfiles/setup.sh