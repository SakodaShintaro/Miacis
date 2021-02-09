cd `dirname $0`

# どこに保存するかの基準位置
root_dir=../..

#
# やねうら王のダウンロード & コンパイル
#
# GitHubからソースコードをダウンロード
git clone https://github.com/yaneurao/YaneuraOu ${root_dir}/Suisho
cd ${root_dir}/Suisho/source
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

# Suisho評価関数のダウンロード
mkdir eval
cd eval
wget https://pjvyqa.bn.files.1drv.com/y4m_dJvlgaZbQFN72YJYe1gQg_hkvKNGYkZjfaGCuVYBKXdFVVD8_sf1TyRk6-ln0waU94nC60BKlfsdMjTcx_sZ7cFC_0594k0-osbqAypam4DGwRN69DghRddCEwq-F5uWzphDLg_u4uxhXeRIL1ZBh1BzAnleilMPOP9I2-XF69-CRYjFG0Mf_FJ9xwWHv4INtldTpuyvtq2BaLIxgGLdw -O Suisho.zip
unzip -q Suisho.zip
mv РЕПа2/eval/nn.bin ./ # 文字化けしてる

# standard_book.dbのダウンロード
mkdir ../book
cd ../book
wget https://github.com/yaneurao/YaneuraOu/releases/download/v4.73_book/standard_book.zip
unzip -q standard_book.zip