import argparse, os, json
import torch
import torchvision as tv
from utils import transforms


def get_opts():
    opt = argparse.Namespace()

    opt.task_name = ''
    opt.exp_name = 'exp1_2stagev2_stage2_seed2'

    #the model reserved position
    opt.fold = "IR_A_seed2"

    #the train data, you need change.
    opt.data_root = '/'

    # choose different case for 4_1,4_2,or4_3
    opt.data_list = 'data/phase2/4_2/'
    opt.exp = '4_2'

    opt.out_root = 'data/opts/'
    opt.out_path = os.path.join(opt.out_root, opt.exp_name, 'fold{fold_n}'.format(fold_n=opt.fold),opt.exp)

    ### Dataloader options ###
    opt.nthreads = 64
    opt.batch_size = 128  # 280
    opt.ngpu = 2

    ### Learning ###
    opt.freeze_epoch = 0
    opt.optimizer_name = 'Adam'
    opt.weight_decay = 0
    opt.lr = 2e-5
    opt.lr_decay_lvl = 0.5
    opt.lr_decay_period = 50
    opt.lr_type = 'cosine_repeat_lr'
    opt.num_epochs = 100

    #when training
    opt.resume = ' '
    opt.evaluate = False
    opt.val_save = False

    #when testing
    # opt.resume = 'model_best.pth'
    # opt.evaluate = True
    # opt.val_save = True

    opt.debug = 0
    ### Other ###
    opt.manual_seed = 2
    opt.log_batch_interval = 10
    opt.log_checkpoint = 10

    #the type of models,need to be changed.
    opt.net_type = 'IR_50DLAS_A'

    opt.pretrained = None
    opt.classifier_type = 'linear'
    opt.loss_type = 'cce'
    opt.fc = True
    opt.alpha_scheduler_type = None
    opt.alpha_scheduler_type = None
    opt.nclasses = 2
    opt.fake_class_weight = 1
    opt.visdom_port = 8097

    opt.git_commit_sha = '3ab79d6c8ec9b280f5fbdd7a8a363a6191fd65ce'
    opt.train_transform = tv.transforms.Compose([
        transforms.MergeItems(True, p=0.1),
        # transforms.LabelSmoothing(eps=0.1, p=0.2),
        transforms.CustomRandomRotation(30, resample=2),
        transforms.CustomResize((125, 125)),
        tv.transforms.RandomApply([
            transforms.CustomCutout(1, 25, 75)], p=0.1),
        # transforms.CustomGaussianBlur(max_kernel_radius=3, p=0.2),
        transforms.CustomRandomResizedCrop(112, scale=(0.5, 1.0)),
        transforms.CustomRandomHorizontalFlip(),
        tv.transforms.RandomApply([
            transforms.CustomColorJitter(0.25, 0.25, 0.25, 0.125)], p=0.2),
        transforms.CustomRandomGrayscale(p=0.1),
        transforms.CustomToTensor(),
        transforms.CustomNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    opt.test_transform = tv.transforms.Compose([
        transforms.CustomResize((125, 125)),
        transforms.CustomRotate(0),
        transforms.CustomRandomHorizontalFlip(p=0),
        transforms.CustomCrop((112, 112), crop_index=0),
        transforms.CustomToTensor(),
        transforms.CustomNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    return opt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--savepath', type=str, default='data/opts/', help='Path to save options')
    conf = parser.parse_args()
    opts = get_opts()
    save_dir = os.path.join(conf.savepath)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    #filename = os.path.join(save_dir, opts.exp_name + '_' + opts.exp + '_' + 'fold{0}'.format(opts.fold) + '_' + opts.task_name + '.opt')
    filename = 'opts/IR_A_seed2_4_2.opt'
    torch.save(opts, filename)
    print('Options file was saved to ' + filename)
