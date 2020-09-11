# どこに保存するかの基準位置($0 = ./の２つ上がMiacisと同階層なのでそこに置く)
root_dir=$(dirname "$0")/../../data

# 棋譜のダウンロード
download_path=${root_dir}/floodgate_kifu
mkdir -p "${download_path}"
wget -P "${download_path}" "http://wdoor.c.u-tokyo.ac.jp/shogi/x/wdoor2015.7z"
wget -P "${download_path}" "http://wdoor.c.u-tokyo.ac.jp/shogi/x/wdoor2016.7z"
wget -P "${download_path}" "http://wdoor.c.u-tokyo.ac.jp/shogi/x/wdoor2018.7z"
wget -P "${download_path}" "http://wdoor.c.u-tokyo.ac.jp/shogi/x/wdoor2019.7z"

# 学習用データ(2016年以降)
train_path=${download_path}/train
mkdir -p "${train_path}"
7z e "${download_path}"/wdoor2016.7z -o"${train_path}"
7z e "${download_path}"/wdoor2017.7z -o"${train_path}"
7z e "${download_path}"/wdoor2018.7z -o"${train_path}"
7z e "${download_path}"/wdoor2019.7z -o"${train_path}"

# 検証用データ(2015年)
valid_path=${download_path}/valid
mkdir -p "${valid_path}"
7z e "${download_path}"/wdoor2015.7z -o"${valid_path}"
