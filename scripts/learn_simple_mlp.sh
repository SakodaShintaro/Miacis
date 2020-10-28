git show -s >git_commit_id.txt
git diff >>git_commit_id.txt
echo -e "learnSimpleMLP\nquit" | ./Miacis_*
~/Miacis/scripts/vsEdax.py --game_num=1000 --level=1 --search_limit=0 --simple_mlp
~/Miacis/scripts/vsEdax.py --game_num=1000 --level=1 --search_limit=10 --simple_mlp
scp -r $(pwd) sakoda:~/learn_result/search_nn/
