import pandas as pd
import numpy as np
import glob
from utils.tools import check


def cefa_main():
    score_list1 = [
        './outputs/C/tta_C_4@1_score.txt',
        './outputs/C_seed2/tta_C_seed2_4@1_score.txt',
        './outputs/IR_A/tta_IR_A_4@1_score.txt',
        './outputs/IR_A_seed2/tta_IR_A_seed2_4@1_score.txt',
        ]
    scores = []
    # loading test names
    fid = open('./data/phase2/4_1/dev_test_list.txt', 'r')
    image_names = ['/'.join(x.strip().split()[0].split('/')[-4:-2]) for i, x in enumerate(fid.readlines()) if i !=0]
    fid.close()
    for i in range(1, 5):
        name_score_dict = {}
        fname = score_list1[i-1]

        sub_score = np.loadtxt(fname, delimiter=',', skiprows=1,usecols=(0,1))
        # choose the first column score
        # sub_score = np.loadtxt(fname, delimiter=',', skiprows=1,usecols=(0))
        sub_score = np.mean(sub_score, axis=1)

        for k, s in zip(image_names, sub_score):
            name_score_dict.setdefault(k, [])
            name_score_dict[k].append(s)

        fout1 = open(fname.replace('score.txt', 'video_score.txt'), 'w')
        for k, v in name_score_dict.items():
            video_score = sum(v) / len(v)
            fout1.write(k + ' ' + str(video_score) + '\n')
        fout1.close()
        print('Aggregated video scores: {}'.format(score_list1[i-1]))
        # cache the `name_score_dict` for fusion
        scores.append(name_score_dict)
    
    # fusing
    fout_1 = open('./scores/4_1/4_1.txt', 'w')
    for k in scores[0]: # the scores from models indexed by `k`
        fused_score = sum([np.mean(x[k]) for x in scores]) / len(scores)
        fout_1.write(k + ' ' + str(fused_score) + '\n')
    fout_1.close()
    print('Fianl fused scores: ./scores/4_1/4_1.txt')
    # check('./outputs/C/C_tta_score.txt')


if __name__ == '__main__':
    cefa_main()
