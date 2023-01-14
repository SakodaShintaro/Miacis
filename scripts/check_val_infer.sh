set -eu

# ライブラリパスへの設定をちゃんとしないと動かない
LD_LIBRARY_PATH=/root/libtorch-1.13.1/lib/

command="checkValInfer
../../data/floodgate_kifu/valid
64
${1}
3200"

echo ${command} | ./Miacis_shogi*