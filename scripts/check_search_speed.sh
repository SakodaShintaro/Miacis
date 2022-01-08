set -eu

command="checkSearchSpeed
setoption name USI_Hash value 8192
setoption name model_name value ${1}
go"

echo ${command} | ./Miacis_shogi*


<< COMMENTOUT
コピペ用
checkSearchSpeed
setoption name USI_Hash value 8192
setoption name model_name value shogi_cat_bl10_ch256.engine
go
COMMENTOUT