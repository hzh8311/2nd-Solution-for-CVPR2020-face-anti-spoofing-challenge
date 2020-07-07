import sys
import argparse,json,random,os
import torch
import torch.backends.cudnn as cudnn
import torchvision as tv
import pandas as pd
import numpy as np
from trainer import Model
from opts import get_opts
import datasets
# from tqdm import tqdm
from collections import OrderedDict
from utils import transforms
from utils.tools import new_valid


def extract_list():

    # Load options
    parser = argparse.ArgumentParser(description='Inference by list')
    parser.add_argument('--config', type = str, help = 'Path to config .opt file. Leave blank if loading from opts.py')
    parser.add_argument('--pth', type = str, help = 'Path to model checkpoint. Leave blank if testing bestmodel')
    # parser.add_argument('--input_list', type = str, help = 'Path to list with image paths')
    parser.add_argument('--output_list', type = str, help = 'Path to list where to store results')
    parser.add_argument('--phase', type=str, default='test', help='')
    parser.add_argument('--tta', type= str, default='', help='Add TTA or not')
    conf = parser.parse_args()

    assert conf.phase in ['valid', 'test'], f'unknown pahse: {conf.phase}'
    opt = torch.load(conf.config) if conf.config else get_opts()
    opt.ngpu = 1
    opt.batch_size=256

    print('===Options==')
    d=vars(opt)
    for k in d.keys():
        print(k,':',d[k])

    if conf.phase == 'valid': # do cross validation
        opt.data_root = '/'
        opt.evaluate = True
        opt.val_save = False
    elif conf.phase == 'test':
        # the dev and test data, you need change.
        opt.data_root = '/'
        opt.evaluate = True
        opt.val_save = True
    else:
        print('unknown phase {}'.format(conf.phase))
        sys.exit(1)

    M = Model(opt)
    ckpt_path = os.path.join(opt.out_path, 'checkpoints', conf.pth)
    checkpoint = torch.load(ckpt_path)

    split = ckpt_path.split('/')[-3][-1] # 4_1/2/3
    print(f'4@{split} Loaded model: {ckpt_path}')


    model_weights = OrderedDict()
    classifier_weights = OrderedDict()
    try:
        for key, value in checkpoint['model_state_dict'].items():
            model_weights[key.replace('module.', '')] = value

        for key, value in checkpoint['classifier_state_dict'].items():
            classifier_weights[key.replace('module.', '')] = value
    except:
        for key, value in checkpoint.items():
            model_weights[key.replace('module.', '')] = value
        for key, value in checkpoint.items():
            classifier_weights[key.replace('module.', '')] = value

    M.model.load_state_dict(model_weights)
    M.model.eval()
    M.classifier.load_state_dict(classifier_weights)
    M.classifier.eval()

    torch.set_grad_enabled(False)
    result_df = pd.DataFrame()

    if conf.tta:
        tta_types = conf.tta.split('_')
    else:
        tta_types = []

    if '5crop' in tta_types:
        crop_idxes = range(5)
    else:
        crop_idxes = [0]

    if 'hflip' in tta_types:
        hflip_probs = [0., 1.]
    else:
        hflip_probs = [0.]

    if 'rotate' in tta_types:
        rotate_angles = [-10, 10]
    else:
        rotate_angles = []

    for crop_idx in crop_idxes:

        crop_transform = transforms.CustomCrop(opt.test_transform.transforms[3].size,
                                               crop_index=crop_idx)
        opt.test_transform.transforms.pop(3)
        opt.test_transform.transforms.insert(3, crop_transform)

        for hflip_prob in hflip_probs:
            hflip_transform = transforms.CustomRandomHorizontalFlip(p=hflip_prob)
            opt.test_transform.transforms.pop(2)
            opt.test_transform.transforms.insert(2, hflip_transform)

            test_loader = datasets.generate_loader(opt, 'val' if conf.phase == 'valid' else 'dev_test', True) # conf.input_list)

            result_arr = np.empty((0,), dtype=np.float32)

            for batch_idx, (rgb_data, depth_data, ir_data, target, dir) in enumerate(test_loader, 1):
                if batch_idx % 200 == 0:
                    print(f'crop: {crop_idx} flip: {hflip_prob} {batch_idx}/' + str(len(test_loader)))
                rgb_data = rgb_data.to(M.device)
                depth_data = depth_data.to(M.device)
                ir_data = ir_data.to(M.device)

                output = M.model(rgb_data, depth_data, ir_data)
                output = M.classifier(output)

                if opt.loss_type == 'bce':
                    output = torch.sigmoid(output)
                else:
                    output = torch.nn.functional.softmax(output, dim=1)[:,1]

                output = output.detach().cpu().numpy()
                result_arr = np.hstack((result_arr, output))

            if conf.phase == 'valid':
                thr, acer = new_valid(result_arr.tolist(), split=int(split)-1)
                print(f'{ckpt_path} Cross Validation ACER: {acer} threshold: {thr}')
                return

            column_name = f'crop{crop_idx}_hflip' if hflip_prob == 1. else f'crop{crop_idx}'
            result_df[column_name] = result_arr
    print ('Crop TTA is done.')

    crop_transform = transforms.CustomCrop(opt.test_transform.transforms[3].size,
                                               crop_index=0)
    opt.test_transform.transforms.pop(3)
    opt.test_transform.transforms.insert(3, crop_transform)
    for rotate_angle in rotate_angles:
        rotate_transform = transforms.CustomRotate(rotate_angle)
        opt.test_transform.transforms.pop(1)
        opt.test_transform.transforms.insert(1, rotate_transform)

        for hflip_prob in hflip_probs:
            hflip_transform = transforms.CustomRandomHorizontalFlip(p=hflip_prob)
            opt.test_transform.transforms.pop(2)
            opt.test_transform.transforms.insert(2, hflip_transform)

            test_loader = datasets.generate_loader(opt, 'val' if conf.phase == 'valid' else 'dev_test', True) # , conf.input_list)

            result_arr = np.empty((0,), dtype=np.float32)

            for batch_idx, (rgb_data, depth_data, ir_data, target, dir) in enumerate(test_loader, 1):
                if batch_idx % 200 == 0:
                    print(f'rotate angle: {rotate_angle} {batch_idx}/' + str(len(test_loader)))

                rgb_data = rgb_data.to(M.device)
                depth_data = depth_data.to(M.device)
                ir_data = ir_data.to(M.device)

                output = M.model(rgb_data, depth_data, ir_data)
                output = M.classifier(output)

                if opt.loss_type == 'bce':
                    output = torch.sigmoid(output)
                else:
                    output = torch.nn.functional.softmax(output, dim=1)[:,1]

                output = output.detach().cpu().numpy()
                result_arr = np.hstack((result_arr, output))

            column_name = f'rotate{rotate_angle}_hflip' if hflip_prob == 1. else f'rotate{rotate_angle}'
            result_df[column_name] = result_arr
    print ('Rotate TTA is done.')

    result_df.to_csv(conf.output_list, index=False)
    print('Extracting done!')

if __name__=='__main__':
    extract_list()
