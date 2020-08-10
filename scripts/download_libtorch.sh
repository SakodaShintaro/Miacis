cd `dirname $0`

# どこに保存するかの基準位置
root_dir=../..

wget -P ${root_dir}/ https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.6.0.zip
unzip -q ${root_dir}/libtorch-cxx11-abi-shared-with-deps-1.6.0.zip -d ${root_dir}/libtorch-tmp
mv ${root_dir}/libtorch-tmp/libtorch ${root_dir}/libtorch-1.6.0
rm ${root_dir}/libtorch-cxx11-abi-shared-with-deps-1.6.0.zip
rmdir ${root_dir}/libtorch-tmp