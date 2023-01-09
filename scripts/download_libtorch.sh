set -eux

cd $(dirname $0)

# どこに保存するかの基準位置
root_dir=../..

version=1.12.0
cuda_version=113
CUDA_STR=cu${cuda_version}
url_file_name=libtorch-cxx11-abi-shared-with-deps-${version}%2B${CUDA_STR}.zip
file_name=libtorch-cxx11-abi-shared-with-deps-${version}+${CUDA_STR}.zip

wget -P ${root_dir}/ https://download.pytorch.org/libtorch/${CUDA_STR}/${url_file_name}
unzip -q ${root_dir}/${file_name} -d ${root_dir}/libtorch-tmp
mv ${root_dir}/libtorch-tmp/libtorch ${root_dir}/libtorch-${version}
rm ${root_dir}/${file_name}
rmdir ${root_dir}/libtorch-tmp