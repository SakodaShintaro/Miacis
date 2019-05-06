# Miacis
MiacisはUSIプロトコルに対応した将棋用思考エンジンです。[将棋所](http://shogidokoro.starfree.jp/)、[将棋GUI](http://shogigui.siganus.com/)などを用いて対局、検討を行うことができます。

内部で深層学習フレームワークとしてPyTorchを利用しています。PyTorchのライセンス規約はNOTICEに、Miacis自身のライセンスはLICENSEに記載しています。

## コンパイル方法
cmake,g++によるコンパイルが可能です。Ubuntu18.04, CUDA10.0, cuDNN7.1の環境で動作することは確認できています。WindosでもVisual Studioに
cmake拡張を導入するなどの方法でコンパイルが可能なのではないかと思われます。

コンパイル時にはPyTorchのC++APIである[LibTorch](https://pytorch.org/get-started/locally/)を必要とします。CMakeLists.txt9行目におけるlibtorchへのパスを適切に設定してください。

以下にLinuxでコンパイルする手順例を示します。

```
#libtorchの取得
wget https://download.pytorch.org/libtorch/nightly/cu100/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip

#Miacisの取得、コンパイル
git clone https://github.com/SakodaShintaro/Miacis
mkdir Miacis/build
cd Miacis/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

正常にコンパイルが進むとbuild以下にMiacis_scalarとMiacis_categoricalの二つプログラムが得られます。前者は状態価値(評価値)をスカラとして一つ出力するモデルであり、AlphaZeroとほぼ同等のアーキテクチャとなります。後者は状態価値(評価値)の確率分布を出力するモデルとなります。

## 対局方法
実行プログラムと同ディレクトリに評価関数パラメータを配置すると思考エンジンとして利用することができます。Miacis_scalarは```torch_sca_bl10_ch64.model```というファイル、Miacis_categorical```はtorch_cat_bl10_ch64.model```というファイルにPyTorchの定める形式でパラメータが格納されている必要があります。bl10はブロック数が10であること、ch64はチャネル数が64であることを示しています。

## 学習方法
Miacisは強化学習による評価関数パラメータの最適化を行うことができます。学習する際にはalphazero_settings.txtを実行プログラムと同ディレクトリに配置しパラメータ等を適切に設定してください。プログラムを実行して「alphaZero」と入力することで現在同ディレクトリに置かれている```torch_cat(sca)_bl10_ch64.model```を初期パラメータとした学習が始まります。

プログラムを実行して「prepareForLearn」を入力するとランダムに初期化したパラメータが```torch_cat(sca)_bl10_ch64.model```という名前で出力されます。ゼロからの学習を試したい方はご活用ください。

学習の際には```torch_cat(sca)_bl10_ch64_before_alphazero.model```という名前で学習前のパラメータが保存されるため、誤って学習を行ってしまった場合はこれをリネームすることで学習前のパラメータに復帰することができます。
