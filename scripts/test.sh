set -eux

# ライブラリパスへの設定をちゃんとしないと動かない
LD_LIBRARY_PATH=/root/libtorch-1.13.1/lib/

# ビルドディレクトリへ移動
SCRIPT_DIR=$(dirname $0)
cd ${SCRIPT_DIR}/../build/

# Make
make -j$(nproc) Miacis_shogi_categorical

# テスト実行
command="usi
setoption name model_name value shogi_cat_transformer_bl12_ch384.ts
isready
testSearch
"
echo ${command} | ./Miacis_shogi_categorical
