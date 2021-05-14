cd $(dirname $0)

# どこに保存するかの基準位置
root_dir=../..

version=1.8.1
url_file_name=libtorch-shared-with-deps-${version}%2Bcu102.zip
file_name=libtorch-cxx11-abi-shared-with-deps-${version}.zip

wget -P ${root_dir}/ https://download.pytorch.org/libtorch/cu102/${url_file_name}
unzip -q ${root_dir}/${file_name} -d ${root_dir}/libtorch-tmp
mv ${root_dir}/libtorch-tmp/libtorch ${root_dir}/libtorch-${version}
rm ${root_dir}/${file_name}
rmdir ${root_dir}/libtorch-tmp
