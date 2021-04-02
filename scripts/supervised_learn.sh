git show -s > git_commit_id.txt
git diff   >> git_commit_id.txt
~/Miacis/scripts/generate_torch_script_model.py
echo -e "supervisedLearn\nquit\n" | ./Miacis_*
scp -r `pwd` sakoda:~/learn_result/supervised/