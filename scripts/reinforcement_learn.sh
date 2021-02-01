git show -s >git_commit_id.txt
git diff >>git_commit_id.txt
echo -e "initParams\nreinforcementLearn\nquit\n" | ./Miacis_*
zip -rq learn_kifu.zip learn_kifu
rm -rf learn_kifu
scp -r $(pwd) sakoda:~/learn_result/reinforcement/
