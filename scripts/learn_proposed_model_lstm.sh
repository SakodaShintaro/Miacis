git show -s >git_commit_id.txt
git diff >>git_commit_id.txt
echo -e "learnProposedModelLSTM\nquit" | ./Miacis_*
~/Miacis/scripts/vsEdax.py --game_num=1000 --level=1 --method_name="proposed_model_lstm" --search_limit=0
~/Miacis/scripts/vsEdax.py --game_num=1000 --level=1 --method_name="proposed_model_lstm" --search_limit=12
scp -r $(pwd) sakoda:~/learn_result/search_nn/
