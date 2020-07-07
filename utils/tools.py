import os
import numpy as np
from sklearn.metrics import confusion_matrix

def check(f):
    with open('./submission/test/mul.txt') as fid:
        keys = [x.strip().split()[0] for x in fid.readlines()]
    with open(f) as fid:
        lines = fid.readlines()
        for i, line in enumerate(lines):
            assert keys[i] == line.split()[0]

def calc_threshold(score, label):
    tol = 1e-2
    best_acer = 100.
    best_acc = 0.
    for thr in np.arange(min(score), max(score)+tol, tol):
        acc = sum([(x > thr) ==y for x,y in zip(score, label)]) / len(label)

        is_best = acc >= best_acc
        if is_best:
            best_acc = acc
            best_thr = thr
    thresholds = np.arange(best_thr-tol**2, best_thr+tol+tol**2, tol**2)
    for thr in thresholds:
        pred_list = [_ > thr for _ in score]
        acc = sum([(x > thr) == y for x, y in zip(score, label)]) / len(label)

        is_best = acc >= best_acc
        if is_best:
            best_acc = acc
            best_thr = thr
    thresholds = np.arange(best_thr-tol**3, best_thr+tol+tol**3, tol**3)
    for thr in thresholds:
        pred_list = [_ > thr for _ in score]
        acc = sum([(x > thr) == y for x, y in zip(score, label)]) / len(label)

        is_best = acc >= best_acc
        if is_best:
            best_acc = acc
            best_thr = thr
        # tn, fp, fn, tp = confusion_matrix(label, pred_list).ravel()
        # apcer = fp/(tn + fp)
        # npcer = fn/(fn + tp)
        # acer = (apcer + npcer)/2
        # is_best = acer <= best_acer
        # if is_best:
        #     best_apcer = apcer
        #     best_npcer = npcer
        #     best_acer = acer
        #     best_thr = thr
    # print(best_thr, best_apcer, best_npcer, best_acer)
    return best_thr


def dev_labels():
    labels = np.loadtxt('data/phase2/4_1/val_list.txt', usecols=[3], skiprows=1, dtype=np.int).tolist()
    return labels


def new_valid(scores, labels=None, split=0):
    acers = []
    # 157691 ../data/phase2/4_1/dev_test_list.txt
    # 155066 ../data/phase2/4_2/dev_test_list.txt
    # 155695 ../data/phase2/4_3/dev_test_list.txt
    if labels is None:
        labels = dev_labels()
    assert len(scores) == len(labels) and len(labels) in [51870, 157690, 155065, 155694], (len(scores), len(labels))
    if len(labels) == 51870: # dev
        scores = [scores[:17068], scores[17068:34761], scores[34761:]]
        labels = [labels[:17068], labels[17068:34761], labels[34761:]]
    # elif len(labels) == 157690: # 4_1
    #     scores = [scores[:17068], scores[157690:157690-17068+34761], scores[157690+155065:157690+155065+51870-34761]]
    #     labels = [labels[:17068], labels[17068:34761], labels[34761:]]
    # elif len(labels) == 155065: # 4_2
    #     scores = [scores[:17068], scores[17068:34761], scores[34761:]]
    #     labels = [labels[:17068], labels[17068:34761], labels[34761:]]
    # elif len(labels) == 157694: # 4_3
    #     scores = [scores[:17068], scores[17068:34761], scores[34761:]]
    #     labels = [labels[:17068], labels[17068:34761], labels[34761:]]

    preds, scores_, labels_ = [], [], []
    others = [x for x in range(3) if x != split]

    thr = calc_threshold(scores[split], labels[split])

    for j in others:
        preds.extend([int(_ > thr) for _ in scores[j]])
        scores_ += scores[j]
        labels_.extend(labels[j])

    tn, fp, fn, tp = confusion_matrix(labels_, preds).ravel()
    apcer = fp/(tn + fp)
    npcer = fn/(fn + tp)
    acer = (apcer + npcer) / 2
    return thr, acer

def fake_test(scores, img_paths):
    scores_dev, scores_test = {}, {}
    for p, s in zip(img_paths, scores):
        vid = '/'.join(p.split('/')[-4:-2])
        if vid.startswith('dev'):
            scores_dev.setdefault(vid, [])
            scores_dev[vid].append(s)
        elif vid.startswith('test'):
            scores_test.setdefault(vid, [])
            scores_test[vid].append(s)
        else:
            print('unknown path', vid)

    flag = 'mean'
    video_scores, video_scores_test = [], []
    for k, v in scores_dev.items():
        final_score = sum(v) / len(v) # get_video_score(v, flag)
        video_scores.append(final_score)
    for k, v in scores_test.items():
        final_score = sum(v) / len(v) # get_video_score(v, flag)
        video_scores_test.append(final_score)
    assert len(video_scores) == 200 and len(video_scores_test) == 2200

    labels_dev = dev_labels()
    labels_test = test_labels()
    thr = calc_threshold(video_scores, labels_dev[args.split])
    pred = [_ > thr for _ in video_scores_test]

    tn, fp, fn, tp = confusion_matrix(labels_test[args.split], pred).ravel()
    apcer = fp/(tn + fp)
    npcer = fn/(fn + tp)
    acer = (apcer + npcer)/2
    return acer

