cp ../../Miacis_categorical .
cp ../../../../setting/alphazero_settings.txt .
git show -s > git_commit_id.txt
echo -e "initParams\nalphaZero\nquit\n" | ./Miacis_categorical
scp -r `pwd` sakoda:~/learn_result/alphazero/