cd `dirname $0`

# どこに保存するかの基準位置
root_dir=../..

#
# やねうら王のダウンロード & コンパイル
#
# GitHubからソースコードをダウンロード
git clone https://github.com/yaneurao/YaneuraOu ${root_dir}/YaneuraOu
cd ${root_dir}/YaneuraOu/source
git checkout b0a3a2a4f7565bbefb85999368df15e9c90c621f

# デフォルトではclangを使うようになっているがg++を使いたいのでMakefileを書き換える
sed -i -e "s/#COMPILER = g++/COMPILER = g++/g" Makefile
sed -i -e "s/COMPILER = clang++/#COMPILER = clang++/g" Makefile

# コンパイル
make -j4

# binディレクトリを作ってそこに動くような環境を作る
mkdir ../bin
cd ../bin
mv ../source/YaneuraOu-by-gcc .

# 評価関数Kristallweizenのダウンロード
mkdir eval
cd eval
git clone https://github.com/Tama4649/Kristallweizen
cp Kristallweizen/Kristallweizen.zip .
unzip -q Kristallweizen.zip

# standard_book.dbのダウンロード
mkdir ../book
cd ../book
wget https://github.com/yaneurao/YaneuraOu/releases/download/v4.73_book/standard_book.zip
unzip -q standard_book.zip

# YaneuraOuと同階層にAyaneも用意
git clone https://github.com/yaneurao/Ayane ../../../Ayane