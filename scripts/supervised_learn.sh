cp ../../Miacis_shogi_scalar .
cp ../../../../setting/supervised_learn_settings.txt .
git show -s > git_commit_id.txt
echo -e "initParams\nsupervisedLearn\nquit\n" | ./Miacis_shogi_scalar
scp -r `pwd` sakoda:~/learn_result/supervised/