set -eu

command="checkSearchSpeed
setoption name USI_Hash value 8192
setoption name model_name value ${1}
go"

echo ${command} | ./Miacis_shogi*