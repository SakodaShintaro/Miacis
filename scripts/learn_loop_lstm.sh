git show -s >git_commit_id.txt
git diff >>git_commit_id.txt
echo -e "learnLoopLSTM\nquit" | ./Miacis_*
~/Miacis/scripts/vsEdax.py --game_num=1000 --level=1 --search_limit=0 --method_name="loop_lstm"
~/Miacis/scripts/vsEdax.py --game_num=1000 --level=1 --search_limit=10 --method_name="loop_lstm"
scp -r $(pwd) sakoda:~/learn_result/search_nn/
