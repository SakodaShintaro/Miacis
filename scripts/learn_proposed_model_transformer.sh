git show -s >git_commit_id.txt
git diff >>git_commit_id.txt
echo -e "learnProposedModelTransformer\nquit" | ./Miacis_*
~/Miacis/scripts/vsEdax.py --game_num=1000 --level=1 --search_limit=0 --proposed_model_transformer
~/Miacis/scripts/vsEdax.py --game_num=1000 --level=1 --search_limit=10 --proposed_model_transformer
scp -r $(pwd) sakoda:~/learn_result/search_nn/
