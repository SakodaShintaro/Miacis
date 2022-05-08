set -eu

# どこに保存するかの基準位置($0 = ./の２つ上がMiacisと同階層なのでそこに置く)
root_dir=$(dirname "$0")/../../data
download_path=${root_dir}/ShogiAIBookData
mkdir -p "${download_path}"

FILE_ID01=17tU75ddw0Ee-Vol-8pW5QveAotv2CMy-
FILE_ID02=1FtmXQlB3D1EYCwYeUnFq4AbodJ6fEVbf
FILE_ID03=1fptlP-1eZt0zxksaRbrZuNmlLW8CAOsw
FILE_ID04=1LIx4QDx_clh-w3JYK5ici4RLZIKaL4qE
FILE_ID05=1CvX5rMG0YMwfF8vLtvwHsyJ68-9IZ0EO
FILE_ID06=1Mmga-eU1c_JQY_HYJsent9V4Z7WI_gSR
FILE_ID07=1swNKlOTwmJZtTnH3Mr07_0F9734QIep8
FILE_ID08=1OLUzX5cWQL5kAZBvpxLZvbWsWI-iriOp
FILE_ID09=193FySmHV_8uUf0uzvOk9ACCpXbRqQ0rK
FILE_ID10=1IDSbsfrPkwgNfBdB7welLcuH9WaMI0xn
FILE_ID11=1ZOvoMWrF-ADBmQoGzNVdZDEt-0MbhXv6
FILE_ID12=1PTXN5PItMERw5wwOMzqpujrJH0J5PJuP
FILE_ID13=1UgN6NB9WP7eIN6SjY5oubl4ret_AeW3X
FILE_ID14=1O2dp48cUbXF3FTPvnBpJjVAWJmfufLWb
FILE_ID15=1XX_qMejwahNpUyip-iNPqJZRdNSmvD6y
FILE_ID16=170CsGz0CuaEUgCIIPpywUP5FZSfwVC6o
FILE_ID17=1p6kyzl0ZEksu1_HGcS5YmAeFtSMtmD9x
FILE_ID18=1I-f3A6S5x-O9SgoCn5FB_lUVGPSuTTVu
FILE_ID19=17QcBJMkF8ceqT1iyRJzSH149VtRuw2WM
FILE_ID20=1GuFGW_PvcWwchfe4dp6YpbNIxUvd6Fr9
FILE_ID21=1KoGss1_XWbJE98_U__-dVCUPUTmyzQm3
FILE_ID22=1iSRtoN5WbpeZ2M00kXWe3VBVD_EwMg2K
FILE_ID23=1RXWA5pR-loCTsfAyE5slNMzq8R2ktHcA
FILE_ID24=1AnBFFqjdfry-tdvV1Ufcq7f-vH5VL64y

for i in `seq 1 24`
do
  d2i=$(printf "%02d" "${i}")
  d3i=$(printf "%03d" "${i}")
  echo $d3i
  FILE_ID=FILE_ID$d2i
  FILE_ID=${!FILE_ID}
  curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=$FILE_ID" > /dev/null
  CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
  curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=$FILE_ID" -o ${download_path}/suisho3kai-$d3i.hcpe
done
