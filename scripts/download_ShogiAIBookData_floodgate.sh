set -eu

# どこに保存するかの基準位置($0 = ./の２つ上がMiacisと同階層なのでそこに置く)
root_dir=$(dirname "$0")/../../data
download_path=${root_dir}/ShogiAIBookData
mkdir -p "${download_path}"

FILE_ID01=1W1FV3TkygfQPEfzAw-RjO29jCusWlY5q

for i in `seq 1 1`
do
  d2i=$(printf "%02d" "${i}")
  d3i=$(printf "%03d" "${i}")
  echo $d3i
  FILE_ID=FILE_ID$d2i
  FILE_ID=${!FILE_ID}
  curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=$FILE_ID" > /dev/null
  CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
  curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=$FILE_ID" -o ${download_path}/floodgate_2019-2021_r3500-$d3i.hcpe
done
