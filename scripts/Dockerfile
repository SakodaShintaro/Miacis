FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
MAINTAINER Shintaro Sakoda

RUN set -x && \
    : "必要なものをインストール" && \
    apt-get update && \
    apt-get install sudo -y && \
    sudo apt-get install git -y && \
    sudo apt-get install vim -y && \
    sudo apt-get install cmake -y && \
    sudo apt-get install python3 -y && \
    sudo apt-get install python3-pip -y && \
    sudo apt-get install p7zip-full -y && \
    sudo apt-get install wget -y && \
    sudo apt-get install unzip -y && \
    pip3 install natsort && \
    : "日本語の導入" && \
    sudo apt-get install language-pack-ja-base language-pack-ja -y && \
    echo "export LANG='ja_JP.UTF-8'" >> ~/.bashrc && \
    : "Miacisの取得" && \
    cd ~ && \
    git clone https://github.com/SakodaShintaro/Miacis && \
    : "libtorchの取得" && \
    ./Miacis/scripts/download_libtorch.sh && \
    : "ビルド更新スクリプトの準備" && \
    mkdir Miacis/src/cmake-build-release && \
    cd Miacis/src/cmake-build-release && \
    echo "git fetch" > update.sh && \
    echo "git reset --hard origin/master" >> update.sh && \
    echo "cmake -DCMAKE_BUILD_TYPE=Release .." >> update.sh && \
    echo "make -j8" >> update.sh && \
    chmod +x update.sh && \
    ./update.sh && \
    : "dotfilesの取得" && \
    cd ~ && \
    git clone https://github.com/SakodaShintaro/dotfiles && \
    ./dotfiles/setup.sh

WORKDIR /root
