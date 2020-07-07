# Code for CVPR2020 face anti-spoofing challenge

## Environment
python = 3.6.5
pytorch = 1.2.0
torchvision = 0.4.0

We need two 12G GPUs on training and one 12G GPU on testing.

## data preparation
Changing the `opt.data_root` at main.py line 24 and inference.py line 45 to your data path where contains train/dev/test sets.

We use all three dev set for the validation propose, and concatenate the dev list and test list for score calculation, i.e. combine 4@* dev image list and 4@* test image list to dev_test_list.txt at ./data/phase2/4_*, * for 1, 2, 3.

## Training models for subsets

### 4@1
refer to train_4_1.sh
```bash
    python main.py --config opts/C_4_1.opt
    python main.py --config opts/C_seed2_4_1.opt
    python main.py --config opts/IR_A_4_1.opt
    python main.py --config opts/IR_A_seed2_4_1.opt
```

### 4@2
refer to train_4_2.sh
```bash
    python main.py --config opts/C_4_2.opt
    python main.py --config opts/C_seed2_4_2.opt
    python main.py --config opts/IR_A_4_2.opt
    python main.py --config opts/IR_A_seed2_4_2.opt
    python main.py --config opts/B0_4_2.opt
```
### 4@3
refer to train_4_3.sh
```bash
    python main.py --config opts/B_pro_4_3.opt
```

## Test with trained models
4@* image scores => 4@* video scores => (optional, models fusion) 4@* final scores

### Calucating test sets' scores
refer to inference.sh
```bash
    ...
    python inference.py --config opts/C_4_1.opt
    ...
    python inference.py --config opts/C_4_2.opt
    ...
    python inference.py --config opts/B_pro_4_3.opt
    ...
```

### Scores aggregation and fusion
refer to ensemble.sh
```bash
    python ensemble_4_1.py
    python ensemble_4_2.py
    python ensemble_4_3.py
    cat ...4@1.txt ...4@2.txt ...4@3.txt > final_scores.txt
```
