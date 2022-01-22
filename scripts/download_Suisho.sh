set -eux

cd `dirname $0`

# どこに保存するかの基準位置
root_dir=../..

# ディレクトリ作成
mkdir ${root_dir}/Suisho

# 移動
cd ${root_dir}/Suisho

# 評価関数取得
curl -RLo ./suisho5_20211123.halfkp.nnue.cpp.xz https://github.com/mizar/YaneuraOu/releases/download/resource/suisho5_20211123.halfkp.nnue.cpp.xz

#
# やねうら王のダウンロード & コンパイル
#
# GitHubからソースコードをダウンロード
git clone https://github.com/yaneurao/YaneuraOu

# ビルド
cd ./YaneuraOu/source &&\
    xz -cdk ../..//suisho5_20211123.halfkp.nnue.cpp.xz > eval/nnue/embedded_nnue.cpp &&\
    nice make -j$(nproc) YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE TARGET_CPU=AVX2 EVAL_EMBEDDING=ON EXTRA_CPPFLAGS='-DENGINE_OPTIONS="\"option=name=FV_SCALE=type=spin=default=24=min=1=max=128\""' COMPILER=g++ TARGET=./Suisho5-YaneuraOu-tournament-avx2 tournament >& >(tee ./Suisho5-YaneuraOu-tournament-avx2.log)

# 元の位置に戻る
cd ../../

# バイナリを持ってくる
mv ./YaneuraOu/source/Suisho5-YaneuraOu-tournament-avx2 .

# standard_book.dbのダウンロード
mkdir ./book
cd ./book
wget https://github.com/yaneurao/YaneuraOu/releases/download/v4.73_book/standard_book.zip
unzip -q standard_book.zip