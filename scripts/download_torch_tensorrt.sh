set -eux

cd $(dirname $0)

# このスクリプトから見て2つ上に保存する
cd ../..

FILE_NAME=libtorchtrt-v1.1.0-cudnn8.2-tensorrt8.2-cuda11.3-libtorch1.11.0.tar.gz
FILE_URL=https://github.com/pytorch/TensorRT/releases/download/v1.1.0/${FILE_NAME}

wget ${FILE_URL}
tar xfv libtorchtrt-v1.1.0-cudnn8.2-tensorrt8.2-cuda11.3-libtorch1.11.0.tar.gz 
rm ${FILE_NAME}
