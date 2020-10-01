git show -s >git_commit_id.txt
git diff >>git_commit_id.txt
echo -e "learnMCTSNet\nquit" | ./Miacis_*
~/Miacis/scripts/vsEdax.py --game_num=1000 --level=1 --search_limit=0 --mcts_net
~/Miacis/scripts/vsEdax.py --game_num=1000 --level=1 --search_limit=10 --mcts_net
~/Miacis/scripts/vsEdax.py --game_num=1000 --level=1 --search_limit=0 --mcts_net --use_readout_only
~/Miacis/scripts/vsEdax.py --game_num=1000 --level=1 --search_limit=10 --mcts_net --use_readout_only
scp -r $(pwd) sakoda:~/learn_result/search_nn/
