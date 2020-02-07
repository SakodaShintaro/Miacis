cd `dirname $0`

# どこに保存するかの基準位置
root_dir=../..

wget -P ${root_dir}/ https://download.pytorch.org/libtorch/cu100/libtorch-cxx11-abi-shared-with-deps-1.2.0.zip
unzip -q ${root_dir}/libtorch-cxx11-abi-shared-with-deps-1.2.0.zip -d ${root_dir}/
rm ${root_dir}/libtorch-cxx11-abi-shared-with-deps-1.2.0.zip