# Miacis

MiacisはUSIプロトコルに対応した将棋用思考エンジンです。[将棋所](http://shogidokoro.starfree.jp/)、[将棋GUI](http://shogigui.siganus.com/)などを用いて対局、検討を行うことができます。

基本的に[AlphaZero](https://arxiv.org/abs/1712.01815)を模倣したものとなっており、深層強化学習を利用した評価関数を用いてモンテカルロ木探索を行います。

独自の工夫点として、評価値をスカラーではなくカテゴリカル分布を用いて出力します。これは[Categorical DQN](https://arxiv.org/abs/1707.06887)にヒントを得たものとなっています。

## コンパイル方法

コンパイルにはcmakeを利用します。ライブラリとして

* CUDA(cuDNN含む)
* TensorRT

を必要とします。環境構築は複雑なのでDockerを利用することをお勧めします。

## Dockerによる環境構築

Dockerおよび[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)をインストールしてあるUbuntuを前提とします。

1. Dockerfileをダウンロードする
```shell
wget https://raw.githubusercontent.com/SakodaShintaro/Miacis/master/docker/Dockerfile
```

2. miacis_imageというイメージを作成する
```shell
docker build -t miacis_image:latest .
```

3. miacis_imageをもとにmiacis_containerというコンテナを作成する
```shell
docker run --gpus all -it --name miacis_container miacis_image:latest bash
```

4. ビルドする
```shell
mkdir ./Miacis/build/
cd ./Miacis/build/
cmake -DCMAKE_BUILD_TYPE=Release ../src
make -j$(nproc) Miacis_shogi_categorical
```

正常にコンパイルが進むとコンテナ内の```/root/Miacis/build```以下に```Miacis_shogi_categorical```というプログラムが得られます。
