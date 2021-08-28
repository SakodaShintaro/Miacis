cd $(dirname $0)

# どこに保存するかの基準位置
root_dir=../..

version=1.9.0
cuda_version=cu111
url_file_name=libtorch-cxx11-abi-shared-with-deps-${version}%2B${cuda_version}.zip
file_name=libtorch-cxx11-abi-shared-with-deps-${version}+${cuda_version}.zip

wget -P ${root_dir}/ https://download.pytorch.org/libtorch/${cuda_version}/${url_file_name}
unzip -q ${root_dir}/${file_name} -d ${root_dir}/libtorch-tmp
mv ${root_dir}/libtorch-tmp/libtorch ${root_dir}/libtorch-${version}-${cuda_version}
rm ${root_dir}/${file_name}
rmdir ${root_dir}/libtorch-tmp
