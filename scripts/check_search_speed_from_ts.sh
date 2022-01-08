set -eu

~/Miacis/scripts/convert_ts_model_to_onnx.py ${1}

~/Miacis/scripts/convert_onnx_to_engine.sh ${1/.model/.onnx} ../../data/floodgate_kifu/valid

command="checkSearchSpeed
setoption name USI_Hash value 8192
setoption name model_name value ${1/.model/.engine}
go"

echo ${command} | ./Miacis_shogi*