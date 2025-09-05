#!/usr/bin/env python3
"""
比较GT、检测和攻击结果的TP/FN分析脚本
输入：GT目录、检测结果目录、攻击结果目录（都是YOLO格式的txt文件）
输出：每张图像的TP/FN统计信息
"""
import os
import numpy as np
import torch
from tqdm import tqdm
# 导入必要的工具函数
from adversarial_attacks.detectors.yolo2 import utils as yolo2_utils
from adversarial_attacks.physical.tcega.utils import label_filter, truths_length


def load_txt(file_path, target_label):
    """加载YOLO格式的标签文件"""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return torch.zeros([0, 5]).float()

    lb = torch.from_numpy(np.loadtxt(file_path)).float()
    if lb.dim() == 1:
        lb = lb.unsqueeze(0)
    return lb


def val_dets(gt_path, det_path, target_label, iou_thresh=0.5):
    """
    计算单张图像的TP/FN情况
    
    Args:
        gt_boxes: GT框 (N, 5) [class, x, y, w, h]
        det_boxes: 检测框 (M, 6) [class, x, y, w, h, conf]
        target_label: 目标类别
        iou_thresh: IoU阈值
    
    Returns:
        dict: 包含TP/FN统计的字典
    """
        
    # 加载数据
    gt_boxes = load_txt(gt_path, target_label)
    det_boxes = load_txt(det_path, target_label)
    
    # 过滤目标类别的GT框
    gt_filtered = label_filter(gt_boxes, labels=[target_label])
    num_gts = truths_length(gt_filtered)
    gt_boxes_filtered = gt_boxes[:num_gts, 1:]  # 去掉类别，保留坐标
    
    # 过滤目标类别的检测框
    det_target_boxes = []
    for box in det_boxes:
        if box[0].item() == target_label:
            det_target_boxes.append(box[1:]) # 去掉类别，保留坐标

    # 计算检测结果的TP/FN
    gt_used_det = set()
    gt_val = np.full(len(gt_boxes_filtered), False, dtype=bool) if len(gt_boxes_filtered) > 0 else np.array([], dtype=bool)

    for det_box in det_target_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes_filtered):
            if gt_idx in gt_used_det:
                continue

            iou = yolo2_utils.bbox_iou(gt_box, det_box, x1y1x2y2=False)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou > iou_thresh and best_gt_idx != -1:
            gt_used_det.add(best_gt_idx)
            gt_val[best_gt_idx] = True

    return gt_val

def calc_asr(gt_dir, det_dir, attack_dir, target_label):
    # 获取所有文件
    gt_files = set(os.listdir(gt_dir)) if os.path.exists(gt_dir) else set()
    det_files = set(os.listdir(det_dir)) if os.path.exists(det_dir) else set()
    attack_files = set(os.listdir(attack_dir)) if os.path.exists(attack_dir) else set()
    
    # 找到共同的文件
    common_files = gt_files.intersection(det_files).intersection(attack_files)
    common_files = sorted(list(common_files))
    
    print(f"找到 {len(common_files)} 个共同文件")
    print("-" * 80)

    if len(common_files) < len(gt_files):
        print("Debug: common files: ", len(common_files), len(gt_files), len(det_files), len(attack_files))
        print("error")
        exit(0)

    det_success = 0
    attack_success = 0
    for file_idx, filename in tqdm(enumerate(common_files)):
        print(f"val {filename}")
        gt_file = os.path.join(gt_dir, filename)
        det_file = os.path.join(det_dir, filename)
        attack_file = os.path.join(attack_dir, filename)

        gt_val_det = val_dets(gt_file, det_file, target_label=target_label)
        gt_val_attack = val_dets(gt_file, attack_file, target_label=target_label)

        det_success += gt_val_det.sum()
        attack_success += ((gt_val_det == True) * (gt_val_attack==False)).sum()

    asr = attack_success / det_success
    print("det_success: ", det_success)
    print("attack_success", attack_success)
    print("asr: ", asr)

    return asr

def main():
    
    gt_dir = 'dataset/coco2017_car/sub100_padded/labels/val2017'
    det_dir = 'runs/coco2017_car/sub100_padded/val/labels'
    attack_dir = 'runs/coco2017_car/sub100_adv/val/labels'
    target_label = 2  # car类别



if __name__ == "__main__":
    main()
