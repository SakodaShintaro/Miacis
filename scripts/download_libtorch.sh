set -eu
cd $(dirname $0)

# どこに保存するかの基準位置
root_dir=../..

version=1.10.0
url_file_name=libtorch-cxx11-abi-shared-with-deps-${version}%2Bcu113.zip
file_name=libtorch-cxx11-abi-shared-with-deps-${version}+cu113.zip

wget -P ${root_dir}/ https://download.pytorch.org/libtorch/cu113/${url_file_name}
unzip -q ${root_dir}/${file_name} -d ${root_dir}/libtorch-tmp
mv ${root_dir}/libtorch-tmp/libtorch ${root_dir}/libtorch-${version}
rm ${root_dir}/${file_name}
rmdir ${root_dir}/libtorch-tmp
