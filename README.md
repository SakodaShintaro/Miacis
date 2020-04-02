# Miacis
MiacisはUSIプロトコルに対応した将棋用思考エンジンです。[将棋所](http://shogidokoro.starfree.jp/)、[将棋GUI](http://shogigui.siganus.com/)などを用いて対局、検討を行うことができます。

深層学習フレームワークとしてPyTorchを利用しています。PyTorchのライセンス規約はNOTICEに、Miacis自身のライセンスはLICENSEに記載しています。

## コンパイル方法
コンパイル時にはPyTorchのC++APIである[LibTorch](https://pytorch.org/get-started/locally/)を必要とします。下例のようにスクリプトを使うなどしてMiacisと同階層にlibtorchを解凍しておくか、すでにダウンロード済みならばCMakeLists.txt9行目におけるlibtorchへのパスを適切に設定してください。

Ubuntu18.04, CUDA10.0, cuDNN7.1, libtorch1.2(for CUDA10.0)の環境においてcmake3.10.2, g++7.4.0でビルドできることが確認できています。以下にLinuxでコンパイルする手順例を示します。

```
# Miacisの取得
git clone https://github.com/SakodaShintaro/Miacis

# libtorchの取得(Miacisと同階層にlibtorchが解凍される)
Miacis/scripts/download_libtorch.sh

# コンパイル
mkdir Miacis/src/cmake-build-release
cd Miacis/src/cmake-build-release
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

正常にコンパイルが進むと```cmake-build-release```以下に
* Miacis_shogi_scalar
* Miacis_shogi_categorical
* Miacis_othello_scalar
* Miacis_othello_categorical

というプログラムが得られます。```*_scalar```は評価値をスカラとして一つ出力するモデルであり、AlphaZeroとほぼ同等のモデルとなります。```*_categorical```は評価値の確率分布を出力するモデルとなります。

以前はWindows(Visual Studio2017にCMake拡張を入れた環境)でもコンパイル可能ではありましたが、現在可能であるかは未確認です。

## 対局方法
USIオプションで指定するパスに評価関数パラメータを配置すると思考エンジンとして利用することができます。Miacis_scalarは```sca_bl10_ch128.model```というファイル、Miacis_categoricalは```cat_bl10_ch128.model```というファイルにPyTorchの定める形式でパラメータが格納されている必要があります。(bl10はブロック数が10であること、ch128はチャネル数が128であることを示しています。)

## 学習方法
MiacisはAlphaZeroと同様の形式による強化学習で評価関数パラメータの最適化を行うことができます。学習する際にはalphazero_settings.txtを実行プログラムと同ディレクトリに配置しパラメータ等を適切に設定してください。プログラムを実行して「alphaZero」と入力することで現在同ディレクトリに置かれている```cat(sca)_bl10_ch128.model```を初期パラメータとした学習が始まります。

プログラムを実行して「initParams」と入力するとランダムに初期化したパラメータが```cat(sca)_bl10_ch128.model```という名前で出力されます。同名のファイルがすでに存在している場合も上書きされるのでご注意ください。

学習の際には```cat(sca)_bl10_ch128_before_alphazero.model```という名前で学習前のパラメータが保存されるため、誤って学習を行ってしまった場合はこれをリネームすることで学習前のパラメータに復帰することができます。