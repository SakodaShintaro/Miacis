set -eux

# ライブラリパスへの設定をちゃんとしないと動かない
LD_LIBRARY_PATH=/root/libtorch-1.13.1/lib/

# 強化学習実行
command="reinforcementLearn"
echo ${command} | ./Miacis_shogi_categorical
