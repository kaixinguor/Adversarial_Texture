#!/usr/bin/env python3
"""
TCEGA类评估基准测试
"""
import os
import torch
import numpy as np
from scipy.interpolate import interp1d
import fnmatch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from adversarial_attacks.physical import TCEGA
from adversarial_attacks.detectors.yolo2 import utils as yolo2_utils
from adversarial_attacks.physical.tcega.utils import label_filter, truths_length

# 1. 准备评测数据
#     - 原始图片、标签 -> 预处理后的图片、真值
#     - 预处理后的图片 + 检测结果 -> 打上补丁的图片
#     - 打上补丁的图片 + 检测结果
#     - 存成COCO格式

def load_txt(lp, target_label):

    label = []

    if os.path.getsize(lp):       #check to see if label file contains data.
        lb = torch.from_numpy(np.loadtxt(lp)).float()
        if lb.dim() == 1:
            lb = lb.unsqueeze(0)

        # add label filter to truth
        # if target_label is not None:
        #     truths = label_filter(lb, labels=[target_label])
        #     num_gts = truths_length(truths)
        #     truths = truths[:num_gts]
        #     lb = truths
        label.append(lb)
    else:
        label.append(torch.ones([1, 7]).float())

    return label[0]

def calc_ap(gt_dir_test, det_dir_test, target_label, iou_thresh=0.5, num_of_samples=100):
    gt_names = fnmatch.filter(os.listdir(gt_dir_test), '*.txt')
    gt_names.sort()
    
    det_names = fnmatch.filter(os.listdir(det_dir_test), '*.txt')
    det_names.sort()

    assert len(gt_names) == len(det_names)

    aps = []
    positives = []
    total = 0.0
    for file_idx, (gt_name, det_name) in enumerate(zip(gt_names, det_names)):
        assert gt_name == det_name

        gt_path = os.path.join(gt_dir_test, gt_name)
        det_path = os.path.join(det_dir_test, det_name)
      
        truths = load_txt(gt_path, target_label)
        boxes = load_txt(det_path, target_label)

        truths = label_filter(truths, labels=[target_label])
        num_gts = truths_length(truths)
        truths = truths[:num_gts, 1:]
        truths = truths.tolist()
        total = total + num_gts
        
        img_positives = []
        for j in range(len(boxes)):
            if boxes[j][6].item() == target_label:
                best_iou = 0
                best_index = 0

                for ib, box_gt in enumerate(truths):
                    iou = yolo2_utils.bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_iou = iou
                        best_index = ib
                if best_iou > iou_thresh:
                    del truths[best_index]
                    positives.append((boxes[j][4].item(), True))
                    img_positives.append((boxes[j][4].item(), True))
                else:
                    positives.append((boxes[j][4].item(), False))
                    img_positives.append((boxes[j][4].item(), False))

    positives = sorted(positives, key=lambda d: d[0], reverse=True)

    tps = []
    fps = []
    confs = []
    tp_counter = 0
    fp_counter = 0
    for pos in positives:
        if pos[1]:
            tp_counter += 1
        else:
            fp_counter += 1
        tps.append(tp_counter)
        fps.append(fp_counter)
        confs.append(pos[0])

    precision = []
    recall = []
    for tp, fp in zip(tps, fps):
        recall.append(tp / total)
        precision.append(tp / (fp + tp))


    if len(precision) > 1 and len(recall) > 1:
        p = np.array(precision)
        r = np.array(recall)
        p_start = p[np.argmin(r)]
        samples = np.arange(0., 1., 1.0 / num_of_samples)
        interpolated = interp1d(r, p, fill_value=(p_start, 0.), bounds_error=False)(samples)
        avg = sum(interpolated) / len(interpolated)
    elif len(precision) > 0 and len(recall) > 0:
        avg = precision[0] * recall[0]
    else:
        avg = float('nan')

    return precision, recall, avg, confs
             
    
def calc_metrics(method, 
                   img_ori_dir, 
                   lbl_ori_dir, 
                   target_label, 
                   save_dir='./test_results', 
                   do_prepare_data=False):
    from adversarial_attacks.physical.tcega.cfg import get_cfgs
    args, kwargs = get_cfgs('yolov2', method)
    args['max_lab'] = 100
    tcega = TCEGA(method=method, args=args, kwargs=kwargs)

    result_dir = f'./test_results/{method}_car100/yolov2_{method}'

    gt_dir_test = './data/test_lab_%s' % tcega.kwargs['name']
    det_dir_test = os.path.join(result_dir, 'detect_results')
    precision, recall, avg, confs = calc_ap(gt_dir_test, det_dir_test, target_label, iou_thresh=0.5, num_of_samples=100)
    print(f'Original AP: {avg:.4f}')

    gt_dir_test = './data/test_lab_%s' % tcega.kwargs['name']
    attack_gt_dir_test = os.path.join(result_dir, 'attack_results_on_gt')
    precision, recall, avg, confs = calc_ap(gt_dir_test, attack_gt_dir_test, target_label, iou_thresh=0.5, num_of_samples=100)
    print(f'Attack AP: {avg:.4f}')

    gt_dir_test = './data/test_lab_%s' % tcega.kwargs['name']
    attack_det_dir_test = os.path.join(result_dir, 'attack_results_ondet')
    precision, recall, avg, confs = calc_ap(gt_dir_test, attack_det_dir_test, target_label, iou_thresh=0.5, num_of_samples=100)
    print(f'Attack AP: {avg:.4f}')


if __name__ == "__main__":

    img_ori_dir = './dataset/coco2017_car/sub100/images/val2017'
    lbl_ori_dir = './dataset/coco2017_car/sub100/yolo_labels/val2017'
    target_label = 2
    method = "TCA"
    save_dir = f'./test_results/{method}_car100'
    
    calc_metrics(method=method, img_ori_dir=img_ori_dir, lbl_ori_dir=lbl_ori_dir, target_label=target_label,
                    save_dir=save_dir, 
                    do_prepare_data=False)


    