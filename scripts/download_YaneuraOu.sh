cd `dirname $0`

#
# やねうら王のダウンロード & コンパイル
#
# GitHubからソースコードをダウンロード
git clone https://github.com/yaneurao/YaneuraOu ../YaneuraOu
cd ../YaneuraOu/source

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
