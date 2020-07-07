import random
import fileinput
import os
train_chen = './phase2/test_list.txt'
f1 = open(train_chen,'a')

file_dir1='./4@1_dev_res.txt'
f = open(file_dir1)
# import pdb
# pdb.set_trace()
pwd = '/home/intern/cqchen3/datasets/CASIA-CeFA/phase2/'
for line in fileinput.input("4@1_test_res.txt"):
    list = line.split('\n')
    dir1 = pwd + list[0]  +"/profile/"
    s = line.split(' ')
    # import pdb
    # pdb.set_trace()
    for filename in os.listdir(dir1):
        rgb_dir = list[0]  + "/profile/" + filename
        depth_dir = rgb_dir.replace('profile', 'depth')
        ir_dir = rgb_dir.replace('profile', 'ir')
        f1.write(rgb_dir + ' ' + depth_dir + ' ' + ir_dir +' ' + s[1])
        # f1.write("\n")


f1.close()
f.close()
