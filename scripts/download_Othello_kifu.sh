cd `dirname $0`

# どこに保存するかの基準位置
root_dir=../..

#
# 棋譜のダウンロード
#
path=${root_dir}/othello_kifu
mkdir ${path}
for i in `seq -f '%02g' 128`
do
  wget -P ${path} "https://www.skatgame.net/mburo/ggs/game-archive/Othello/Othello.${i}e4.ggf.bz2"
  bzip2 -d ${path}/Othello.${i}e4.ggf.bz2
done
