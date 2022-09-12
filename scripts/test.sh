set -eux

SCRIPT_DIR=$(dirname $0)

cd ${SCRIPT_DIR}/../build/

make -j$(nproc) Miacis_shogi_categorical

command="usi
setoption name model_name value shogi_cat_bl10_ch256.ts
isready
"

echo ${command} | ./Miacis_shogi_categorical
