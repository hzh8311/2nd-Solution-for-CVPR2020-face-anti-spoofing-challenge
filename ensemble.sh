# the environment for ourself. if you need, you can change it.
# source /home/cqchen3/.bashrc_pytorch

python ensemble_4_1.py
python ensemble_4_2.py
python ensemble_4_3.py
cat ./scores/4_1/4_1.txt ./scores/4_2/4_2.txt ./scores/4_2/4_2.txt > final_scores.txt