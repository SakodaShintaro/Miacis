set -eu

command="checkValInferHcpe
../../data/ShogiAIBookData/floodgate_2019-2021_r3500-001.hcpe
64
${1}"

echo ${command} | ./Miacis_shogi*