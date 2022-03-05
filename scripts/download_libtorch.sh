cd $(dirname $0)

# どこに保存するかの基準位置
root_dir=../..

version=1.10.1
CUDA_VERSION=113
url_file_name=libtorch-cxx11-abi-shared-with-deps-${version}%2Bcu${CUDA_VERSION}.zip
file_name=libtorch-cxx11-abi-shared-with-deps-${version}+cu${CUDA_VERSION}.zip

wget -P ${root_dir}/ https://download.pytorch.org/libtorch/cu${CUDA_VERSION}/${url_file_name}
unzip -q ${root_dir}/${file_name} -d ${root_dir}/libtorch-tmp
mv ${root_dir}/libtorch-tmp/libtorch ${root_dir}/libtorch-${version}-cu${CUDA_VERSION}
rm ${root_dir}/${file_name}
rmdir ${root_dir}/libtorch-tmp
