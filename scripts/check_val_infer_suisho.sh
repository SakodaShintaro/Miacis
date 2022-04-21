set -eu

command="checkValInferSuisho
../../data/ShogiAIBookData_Suisho/suisho3kai-001.hcpe
64
${1}"

# command="checkValInferSuisho
# ../../data/ShogiAIBookData/dlshogi_with_gct-001.hcpe
# 64
# ${1}"

echo ${command} | ./Miacis_shogi*