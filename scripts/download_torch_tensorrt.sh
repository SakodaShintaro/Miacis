set -eux

cd $(dirname $0)

# このスクリプトから見て2つ上に保存する
cd ../..

VERSION=1.2.0
FILE_NAME=libtorchtrt-${VERSION}-cudnn8.4-tensorrt8.4-cuda11.6-libtorch1.12.1-x86_64-linux.tar.gz
FILE_URL=https://github.com/pytorch/TensorRT/releases/download/v${VERSION}/${FILE_NAME}
wget ${FILE_URL}
tar xfv ${FILE_NAME}
rm ${FILE_NAME}
mv torch_tensorrt torch_tensorrt-${VERSION}
