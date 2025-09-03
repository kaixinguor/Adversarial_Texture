#!/usr/bin/env python3
"""
TCEGA类评估基准测试
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from adversarial_attacks.physical import TCEGA
from adversarial_attacks.utils.aux_tool import set_random_seed
from adversarial_attacks.detectors.yolo2 import load_data
from adversarial_attacks.detectors.yolo2 import utils as yolo2_utils
from adversarial_attacks.physical.tcega.utils import label_filter, truths_length
from adversarial_attacks.utils.aux_tool import unloader

def prepare_data(attacker, img_ori_dir, lbl_ori_dir, target_label):

    conf_thresh = 0.5
    nms_thresh = 0.4
    img_dir = './data/test_padded'
    lab_dir = './data/test_lab_%s' % attacker.kwargs['name']
    det_dir = './data/test_det_%s' % attacker.kwargs['name'] # 

    print("remove old data")
    if os.path.exists(lab_dir):
        shutil.rmtree(lab_dir)
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if os.path.exists(det_dir):
        shutil.rmtree(det_dir)

    if lab_dir is not None:
        if not os.path.exists(lab_dir):
            os.makedirs(lab_dir)
    if img_dir is not None:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
    if det_dir is not None:
        if not os.path.exists(det_dir):
            os.makedirs(det_dir)
    print('preparing the test data')

    data_nl = load_data.InriaDataset(img_ori_dir, lbl_ori_dir, attacker.args['max_lab'], attacker.args.img_size, target_label=target_label, shuffle=False)
    loader_nl = torch.utils.data.DataLoader(data_nl, batch_size=1, shuffle=False, num_workers=10)
    with torch.no_grad():
        for batch_idx, (img_batch, lab_batch, img_path_batch, lab_path_batch) in tqdm(enumerate(loader_nl), total=len(loader_nl)):
            img_batch = img_batch.to(attacker.device)
            det_output = attacker.model(img_batch)
            all_det_boxes = yolo2_utils.get_region_boxes_general(det_output, attacker.model, 
                                                    conf_thresh, attacker.kwargs['name'])
            
            for i in range(img_batch.size(0)):
                boxes = lab_batch[i].detach().cpu().numpy()

                if lab_dir is not None:
                    filename = os.path.basename(lab_path_batch[i])
                    save_path = os.path.join(lab_dir, filename)
                    np.savetxt(save_path, boxes, fmt='%f')
                    
                if img_dir is not None:
                    filename = os.path.basename(img_path_batch[i]).replace('.jpg', '.png')
                    save_path = os.path.join(img_dir, filename)
                    img = unloader(img_batch[i].detach().cpu())
                    img.save(save_path)

                # 保存用于攻击的检测目标
                det_boxes = all_det_boxes[i]
                det_boxes = yolo2_utils.nms(det_boxes, nms_thresh)
                new_det_boxes = det_boxes[:, [6, 0, 1, 2, 3]]
                new_det_boxes = new_det_boxes[new_det_boxes[:, 0] == target_label]
                new_det_boxes = new_det_boxes.detach().cpu().numpy()
                if det_dir is not None:
                    filename = os.path.basename(lab_path_batch[i])
                    save_path = os.path.join(det_dir, filename)
                    np.savetxt(save_path, new_det_boxes, fmt='%f')
                
    print('preparing done')

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

def test_model(attacker, loader, target_label, conf_thresh=0.5, nms_thresh=0.4, iou_thresh=0.5, num_of_samples=100,
                old_fasion=True, do_attack=True, save_dir=None, gt_dir=None):
    """测试模型性能"""
    attacker.model.eval()
    total = 0.0
    proposals = 0.0
    correct = 0.0
    batch_num = len(loader)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        positives = []
        for batch_idx, (data, target, img_paths, lab_paths) in tqdm(enumerate(loader), total=batch_num, position=0):
            data = data.to(attacker.device)

            if do_attack:
                target = target.to(attacker.device)

                adv_patch = attacker.generate_adv_patch()
                adv_batch_t = attacker.patch_transformer(adv_patch, target, target_label, attacker.args.img_size, 
                                                    do_rotate=True, rand_loc=False,
                                                    pooling=attacker.args.pooling, 
                                                    old_fasion=old_fasion)
                data = attacker.patch_applier(data, adv_batch_t)
            
            output = attacker.model(data)
            all_boxes = yolo2_utils.get_region_boxes_general(output, attacker.model, 
                                                    conf_thresh, attacker.kwargs['name'] if attacker.kwargs else 'yolov2')
            
            for i in range(len(all_boxes)):
                boxes = all_boxes[i]
                boxes = yolo2_utils.nms(boxes, nms_thresh)
                if save_dir is not None:
                    filename = os.path.basename(lab_paths[i])
                    save_path = os.path.join(save_dir, filename)
                    np.savetxt(save_path, boxes, fmt='%f')

                if gt_dir is not None:
                    gt_filename = os.path.basename(lab_paths[i])
                    
                    gt_path = os.path.join(gt_dir, gt_filename)
                    truths = load_txt(gt_path, target_label)
                else:
                    truths = target[i].view(-1, 5)

                truths = label_filter(truths, labels=[target_label])
                num_gts = truths_length(truths)
                truths = truths[:num_gts, 1:]
                truths = truths.tolist()
                total = total + num_gts
                
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
                        else:
                            positives.append((boxes[j][4].item(), False))
                            
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
    
def run_evaluation(method, 
                   img_ori_dir, 
                   lbl_ori_dir, 
                   target_label, 
                   save_dir='./test_results', 
                   do_prepare_data=False):
    from adversarial_attacks.physical.tcega.cfg import get_cfgs
    args, kwargs = get_cfgs('yolov2', method)
    args['max_lab'] = 100
    tcega = TCEGA(method=method, args=args, kwargs=kwargs)

    """运行完整的评估流程"""
    if not hasattr(tcega, 'test_cloth'):
        raise ValueError("Please load pretrained attack model first using load_pretrained_attack()")
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, f'yolov2_{method}')
    os.makedirs(save_path, exist_ok=True)

    if do_prepare_data:
        prepare_data(tcega, img_ori_dir, lbl_ori_dir, target_label)
    
    # 创建测试数据加载器
    img_dir_test = './data/test_padded'
    lab_dir_test = f'./data/test_lab_{tcega.kwargs["name"] if tcega.kwargs else "yolov2"}'
    det_dir_test = f'./data/test_det_{tcega.kwargs["name"] if tcega.kwargs else "yolov2"}'

    test_data = load_data.InriaDataset(img_dir_test,
                                        lab_dir_test, 
                                        tcega.kwargs['max_lab'], 
                                        tcega.args.img_size, 
                                        shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, tcega.args.batch_size, shuffle=False, num_workers=10)
    
    # 运行测试
    plt.figure(figsize=[15, 10])

    print("Running detection")
    prec_ori, rec_ori, ap_ori, confs_ori = test_model(tcega, test_loader, target_label, conf_thresh=0.01, 
            old_fasion=tcega.kwargs['old_fasion'] if tcega.kwargs else True, do_attack=False,
            save_dir=os.path.join(save_path, 'detect_results'),
            gt_dir=lab_dir_test)
    
    print("Running attack on ground truth")
    prec_attack_gt, rec_attack_gt, ap_attack_gt, confs_attack_gt = test_model(tcega, test_loader, target_label, conf_thresh=0.01, 
            old_fasion=tcega.kwargs['old_fasion'] if tcega.kwargs else True, do_attack=True,
            save_dir=os.path.join(save_path, 'attack_results_on_gt'),
            gt_dir=lab_dir_test)
    
    print("Running attack on detection results")
    attack_data = load_data.InriaDataset(img_dir_test,
                                        det_dir_test, 
                                        tcega.kwargs['max_lab'], 
                                        tcega.args.img_size, 
                                        shuffle=False)
    attack_loader = torch.utils.data.DataLoader(attack_data, tcega.args.batch_size, shuffle=False, num_workers=10)
    prec_attack_det, rec_attack_det, ap_attack_det, confs_attack_det = test_model(tcega, attack_loader, target_label, conf_thresh=0.01, 
            old_fasion=tcega.kwargs['old_fasion'] if tcega.kwargs else True, do_attack=True,
            save_dir=os.path.join(save_path, 'attack_results_ondet'),
            gt_dir=lab_dir_test)
    
    # 保存结果
    np.savez(save_path, prec_ori=prec_ori, rec_ori=rec_ori, ap_ori=ap_ori, confs_ori=confs_ori,
            prec_attack_gt=prec_attack_gt, rec_attack_gt=rec_attack_gt, ap_attack_gt=ap_attack_gt, confs_attack_gt=confs_attack_gt,
            prec_attack_det=prec_attack_det, rec_attack_det=rec_attack_det, ap_attack_det=ap_attack_det, confs_attack_det=confs_attack_det,
            adv_patch=tcega.cloth.detach().cpu().numpy())
    print(f'Original AP: {ap_ori:.4f}, Attack AP: {ap_attack_gt:.4f}, Attack on Detection AP: {ap_attack_det:.4f}')
    
    # 绘制攻击前后的PR曲线对比
    plt.plot(rec_ori, prec_ori, 'b-', linewidth=2, label=f'Original: AP {ap_ori:.3f}')
    plt.plot(rec_attack_gt, prec_attack_gt, 'r-', linewidth=2, label=f'Attack on GT: AP {ap_attack_gt:.3f}')
    plt.plot(rec_attack_det, prec_attack_det, 'g-', linewidth=2, label=f'Attack on Detection: AP {ap_attack_det:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{method} Attack Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path + '_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存对抗补丁图像
    unloader(tcega.cloth[0]).save(save_path + '.png')
    
    return prec_attack_gt, rec_attack_gt, ap_attack_gt, confs_attack_gt, prec_attack_det, rec_attack_det, ap_attack_det, confs_attack_det

if __name__ == "__main__":
    # test_basic_function()

    # img_ori_dir = './data/INRIAPerson/Test/pos'
    # lbl_ori_dir = './data/train_labels'
    # target_label = 0
    # method = "TCA"

    # img_ori_dir = './dataset/coco2017_person/sub100/images/val2017'
    # lbl_ori_dir = './dataset/coco2017_person/sub100/yolo_labels/val2017'
    # target_label = 0
    # method = "TCA"
    # save_dir = f'./test_results/{method}_person100'

    img_ori_dir = './dataset/coco2017_car/sub100/images/val2017'
    lbl_ori_dir = './dataset/coco2017_car/sub100/yolo_labels/val2017'
    target_label = 2
    method = "TCA"
    save_dir = f'./test_results/{method}_car100'

    set_random_seed()
    
    run_evaluation(method=method, img_ori_dir=img_ori_dir, lbl_ori_dir=lbl_ori_dir, target_label=target_label,
                    save_dir=save_dir, 
                    do_prepare_data=True)

    # 评测四种方法
    # for method_idx, method in enumerate(['RCA', 'TCA', 'EGA', 'TCEGA']):
    #     set_random_seed()
    #     preproduce_benchmark(method, img_ori_dir, prepare_data=False)