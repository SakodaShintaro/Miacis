git show -s > git_commit_id.txt
git diff   >> git_commit_id.txt
echo -e "initParams\nsupervisedLearn\nquit\n" | ./Miacis_*
scp -r `pwd` sakoda:~/learn_result/supervised/