cd `dirname $0`

# どこに保存するかの基準位置
root_dir=../..

#
# Edaxのダウンロード
#
mkdir ${root_dir}/Edax
wget -P ${root_dir}/Edax https://github.com/abulmo/edax-reversi/releases/download/v4.4/edax-linux.7z
7z x -o${root_dir}/Edax ${root_dir}/Edax/edax-linux.7z
wget -P ${root_dir}/Edax https://github.com/abulmo/edax-reversi/releases/download/v4.4/eval.7z
7z x -o${root_dir}/Edax ${root_dir}/Edax/eval.7z
