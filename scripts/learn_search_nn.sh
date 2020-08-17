git show -s > git_commit_id.txt
git diff   >> git_commit_id.txt
echo -e "learnMCTSNet\nlearnProposedModel\nlearnStackedLSTM\nquit" | ./Miacis_*
scp -r `pwd` sakoda:~/learn_result/search_nn/