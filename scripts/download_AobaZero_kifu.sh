#!/bin/bash

# どこに保存するかの基準位置($0 = ./の２つ上がMiacisと同階層なのでそこに置く)
root_dir=$(dirname "$0")/../../data/aobazero_kifu
mkdir -p "${root_dir}/train"

THRESHOLD=000014300000

while read row; do
  file_name=$(echo ${row} | cut -d , -f 1)
  file_url=$(echo ${row} | cut -d , -f 2)

  # 改行削除
  file_url=$(echo ${file_url} | sed -e "s/[\r\n]\+//g")

  # file_urlは
  # https://drive.google.com/file/d/1qQG2LgIvzShnfDM8be5cxGIg6g3mrSt2/view?usp=drivesdk
  # のような感じで取得できる
  # このうちID部分だけを抽出
  file_id=${file_url%/*}
  file_id=${file_id##*/}

  echo "${file_name} : ${file_id}"

  file_number=${file_name//[^0-9]/}

  # THRESHOLDより大きいものだけをダウンロード
  # 2重にダウンロードしないように存在判定を入れる
  if [ "${file_number}" -ge ${THRESHOLD} ] && [ ! -f "${root_dir}/${file_name}" ] ; then
    # ダウンロード
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" >/dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${file_id}" -o "${root_dir}/${file_name}"

    # 解凍
    # 教師あり学習の実装の都合上,*.csa.xzなのをdata/archXXXX/*.csaとして保存
    save_dir="${root_dir}/data/${file_name%%.*}"
    mkdir -p "${save_dir}"
    xz -dc "${root_dir}/${file_name}" >"${save_dir}/${file_name%.*}"
  else
    echo "スキップ"
  fi
done <"$(dirname "$0")/AobaZero_kifuID.csv"
