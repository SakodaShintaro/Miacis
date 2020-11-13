git show -s >git_commit_id.txt
git diff >>git_commit_id.txt
echo -e "learnSimpleMLP\nquit" | ./Miacis_*
~/Miacis/scripts/vsEdax.py --game_num=1000 --level=1 --method_name="simple_mlp" --search_limit=0
scp -r $(pwd) sakoda:~/learn_result/search_nn/
