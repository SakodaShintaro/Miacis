cd `dirname $0`

# どこに保存するかの基準位置
root_dir=../..

#
# 棋譜のダウンロード
#
train_path=${root_dir}/othello_train_kifu
mkdir ${train_path}
for i in `seq -f '%02g' 1 100`
do
  wget -P ${train_path} "https://www.skatgame.net/mburo/ggs/game-archive/Othello/Othello.${i}e4.ggf.bz2"
  bzip2 -d ${train_path}/Othello.${i}e4.ggf.bz2
done

valid_path=${root_dir}/othello_valid_kifu
mkdir ${valid_path}
for i in `seq -f '%02g' 101 128`
do
  wget -P ${valid_path} "https://www.skatgame.net/mburo/ggs/game-archive/Othello/Othello.${i}e4.ggf.bz2"
  bzip2 -d ${valid_path}/Othello.${i}e4.ggf.bz2
done
