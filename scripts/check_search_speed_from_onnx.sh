set -eu

SCRIPT_DIR=$(dirname $0)

~/Miacis/scripts/convert_onnx_to_engine.sh ${1} ${SCRIPT_DIR}/../../data/floodgate_kifu/valid

command="checkSearchSpeed
setoption name USI_Hash value 8192
setoption name model_name value ${1/.onnx/.engine}
go"

echo ${command} | ./Miacis_shogi*