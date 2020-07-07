import random
import fileinput
train_chen = './phase2/train_list.txt'
f1 = open(train_chen,'a')
# dev_chen = './phase2/val_list.txt'
# f2 = open(dev_chen,'a')

file_dir1='./4@1_train.txt'
f = open(file_dir1)

# train/val split
# mid = []
# for line in fileinput.input("4@1_train.txt"):
#     list = line.split('/')
#     list1 = list[1].split('_')[1]
#     mid.append(list1)
# subjects = set(mid)
# subs = random.sample(subjects,20)
# for line in fileinput.input("4@1_train.txt"):
#     list = line.split('/')
#     list1 = list[1].split('_')[1]
#     list2 = line.split(' ')
#
#     s1 = line.split(' ')
#         # import pdb
#         # pdb.set_trace()
#     depth_dir = s1[0].replace('profile', 'depth')
#     ir_dir = s1[0].replace('profile', 'ir')
#
#
#     if(list1 in subs):
#         f2.write(s1[0] + ' ' + depth_dir + ' ' + ir_dir + ' ' + s1[1])
#     else:
#         f1.write(s1[0] + ' ' + depth_dir + ' ' + ir_dir + ' ' + s1[1])
#
# f1.close()
# f2.close()
# f.close()

for line in fileinput.input("4@1_train.txt"):
    s1 = line.split(' ')
    depth_dir = s1[0].replace('profile', 'depth')
    ir_dir = s1[0].replace('profile', 'ir')
    f1.write(s1[0] + ' ' + depth_dir + ' ' + ir_dir + ' ' + s1[1])

# for line in fileinput.input("4@123_dev.txt"):
#     s1 = line.split(' ')
#     print(s1)
#     print(s1[1])
#     depth_dir = s1[0].replace('profile', 'depth')
#     ir_dir = s1[0].replace('profile', 'ir')
#     f2.write(s1[0] + ' ' + depth_dir + ' ' + ir_dir + ' ' + s1[1])

f1.close()
# f2.close()
f.close()

