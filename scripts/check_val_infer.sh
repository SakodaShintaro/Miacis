set -eu

command="checkValInfer
../../data/floodgate_kifu/valid
64
${1}
3200"

echo ${command} | ./Miacis_shogi*