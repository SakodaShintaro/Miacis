# Miacis

MiacisはUSIプロトコルに対応した将棋用思考エンジンです。[将棋所](http://shogidokoro.starfree.jp/)、[将棋GUI](http://shogigui.siganus.com/)などを用いて対局、検討を行うことができます。

基本的に[AlphaZero](https://arxiv.org/abs/1712.01815)を模倣したものとなっており、深層強化学習を利用した評価関数を用いてモンテカルロ木探索を行います。

独自の工夫点として、評価値をスカラーではなくカテゴリカル分布を用いて出力します。これは[Categorical DQN](https://arxiv.org/abs/1707.06887)にヒントを得たものとなっています。

## コンパイル方法

コンパイルにはcmakeを利用します。ライブラリとして

* CUDA(cuDNN含む)
* LibTorch
* TensorRT
* TRTorch

を必要とします。環境構築は複雑なのでDockerを利用することをお勧めします。

## Dockerによる環境構築

Dockerおよび[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)をインストールしてあるUbuntuを前提とします。
以下に

1. Dockerfileをダウンロードする
2. miacis_trtorchというイメージをタグをpytorch1.7-cuda11.1-trt7.2.1として作成する
3. miacis_trtorchというイメージをもとにmiacis_containerという名前でコンテナを作成する

を行うコマンドを示します。適当な空のディレクトリ内で実行してください。

```shell
wget https://raw.githubusercontent.com/SakodaShintaro/Miacis/master/scripts/Dockerfile
docker build -t miacis_trtorch:pytorch1.7-cuda11.1-trt7.2.1 .
docker run --gpus all -it --name miacis_container miacis_trtorch:pytorch1.7-cuda11.1-trt7.2.1 bash
```

正常にコンパイルが進むとコンテナ内の```/root/Miacis/src/cmake-build-release```以下に```Miacis_shogi_categorical```というプログラムが得られます。

## 対局方法

USIオプション```model_name```で指定するパスに評価関数パラメータを配置すると思考エンジンとして利用することができます。デフォルトでは```Miacis_shogi_categorical```
と同じディレクトリに```shogi_cat_bl10_ch256.model```というファイルにパラメータが格納されている必要があります。(bl10はブロック数が10であること、ch256はチャネル数が256であることを示しています。)