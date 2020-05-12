git show -s > git_commit_id.txt
git diff   >> git_commit_id.txt
echo -e "initParams\nalphaZero\nquit\n" | ./Miacis_othello_*
~/Miacis/scripts/vsEdax.py --exp_search
scp -r `pwd` sakoda:~/learn_result/alphazero/othello/