#!/bin/bash
# the environment for ourself. if you need, you can change it.
# source /home/cqchen3/.bashrc_pytorch

# ==========================CASIA CeFA=======================
#the config(Path to config .opt file.) and output_list(Path to list where to store results) need to be change.

config="opts/C_4_1.opt"
output_list="outputs/C/tta_C_4@1_score.txt"

# config="opts/C_seed2_4_1.opt"
# output_list="outputs/C_seed2/tta_C_seed2_4@1_score.txt"

# config="opts/IR_A_4_1.opt"
# output_list="outputs/IR_A/tta_IR_A_4@1_score.txt"

# config="opts/IR_A_seed2_4_1.opt"
# output_list="outputs/IR_A_seed2/tta_IR_A_seed2_4@1_score.txt"

# config="opts/C_4_2.opt"
# output_list="outputs/C/tta_C_4@2_score.txt"

# config="opts/C_seed2_4_2.opt"
# output_list="outputs/C_seed2/tta_C_seed2_4@2_score.txt"

# config="opts/IR_A_4_2.opt"
# output_list="outputs/IR_A/tta_IR_A_4@2_score.txt"

# config="opts/IR_A_seed2_4_2.opt"
# output_list="outputs/IR_A_seed2/tta_IR_A_seed2_4@2_score.txt"

# config="opts/B0_4_2.opt"
# output_list="outputs/BO/tta_B0_4@2_score.txt"

# config="opts/B_pro_4_3.opt"
# output_list="outputs/B_pro/tta_B_pro_4@3_score.txt"

pth="model_best.pth"
python inference.py --config $config --pth $pth --output_list $output_list --tta "hflip"
