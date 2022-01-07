set -eu

# どこに保存するかの基準位置($0 = ./の２つ上がMiacisと同階層なのでそこに置く)
root_dir=$(dirname "$0")/../../data
download_path=${root_dir}/ShogiAIBookData
mkdir -p "${download_path}"

FILE_ID01=1BAviO-7PzhdGp8BJkjTulMry9_braU6N
FILE_ID02=1cazdNgSJqSC6dvHibKpBQvYaZdconukm
FILE_ID03=1gEKouO02A1a6qzxgdo8GbpUnZMbP1KNp
FILE_ID04=1hIX5MScjUwofp9FF_64K9SEnqajASXGw
FILE_ID05=1i-ieqBOpqRXTeKVjk5GaOY0BTV26glUe
FILE_ID06=10KAAOrMJ50QfaC8T70JLSL2ht9XY_XaK
FILE_ID07=1OEXgTtSCiEvHFUWZl1T9DpotT5MirPMz
FILE_ID08=1PrNC3IkB_Nziv3uIdeXKL5FfEzhCP4he
FILE_ID09=1BYdGJF6J7jNxFTuZjcHCcZBuHtOqwla8
FILE_ID10=1ttdHuowjAt8EkkhLkkb3ykEEytHP6j7U
FILE_ID11=16eavhB00-5Sni61XQlvPFz4BBgVXKxDW
FILE_ID12=1MNB01V0OOoZf0wmyZWHRy63ZR9IfJbkR
FILE_ID13=1DrRnZAUanKg1HB_dYps5FpaN7ETHop2W
FILE_ID14=1N2w88pFz6vzJ178DaMvuVo_zFnTYZAzM
FILE_ID15=1Hjx8xYm_Q6ISBRabDMP_8aULKnymtxXC
FILE_ID16=12KI7Zy0AzBZq7-Q0FBS3gNazs7GoKKHO
FILE_ID17=1p26zp5ZC_BN7VpslXg_Ey1PwONZR2aOR
FILE_ID18=18SZ1hlypPtvsi0IJ-1rEJCuC-VpH9e6a
FILE_ID19=1Y12ZFJnSaR_SxgEXqZhOlL6FYZm47nKY
FILE_ID20=1L9u9T3KSoRfGvpUnK1EhGHargk6XO1RM
FILE_ID21=16_o6IBb8uJQY-qcQwErQprTNnr4k3bpc
FILE_ID22=1uG67ZN5mpYpSnY4nTS2XXVbDpJlia_4V
FILE_ID23=1D4DDt6st-SHgilpF8tT3ajXKEdpt6e_I
FILE_ID24=1l9qMnC34HX4z3wnlt9cAhDgEXRtrufTd

for i in `seq 1 24`
do
  d2i=$(printf "%02d" "${i}")
  d3i=$(printf "%03d" "${i}")
  echo $d3i
  FILE_ID=FILE_ID$d2i
  FILE_ID=${!FILE_ID}
  curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=$FILE_ID" > /dev/null
  CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
  curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=$FILE_ID" -o ${download_path}/dlshogi_with_gct-$d3i.hcpe
done
