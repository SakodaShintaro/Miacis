git show -s >git_commit_id.txt
git diff >>git_commit_id.txt
~/Miacis/scripts/generate_torch_script_model.py
echo -e "reinforcementLearn\nquit\n" | ./Miacis_*
zip -rq learn_kifu.zip learn_kifu
rm -rf learn_kifu
