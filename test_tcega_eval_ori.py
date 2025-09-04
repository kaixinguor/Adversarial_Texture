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

def prepare_data(attacker, img_ori_dir, target_label):

    conf_thresh = 0.5
    nms_thresh = 0.4
    img_dir = './data/test_padded'
    lab_dir = './data/test_lab_%s' % attacker.kwargs['name']

    print("remove old data")
    if os.path.exists(lab_dir):
        shutil.rmtree(lab_dir)
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)

    data_nl = load_data.InriaDataset(img_ori_dir, None, attacker.kwargs['max_lab'], attacker.args.img_size, shuffle=False)
    loader_nl = torch.utils.data.DataLoader(data_nl, batch_size=attacker.args.batch_size, shuffle=False, num_workers=10)
    if lab_dir is not None:
        if not os.path.exists(lab_dir):
            os.makedirs(lab_dir)
    if img_dir is not None:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
    print('preparing the test data')
    with torch.no_grad():
        for batch_idx, (data, labs) in tqdm(enumerate(loader_nl), total=len(loader_nl)):
            data = data.to(attacker.device)
            output = attacker.model(data)
            all_boxes = yolo2_utils.get_region_boxes_general(output, attacker.model, conf_thresh, attacker.kwargs['name'])
            for i in range(data.size(0)):
                boxes = all_boxes[i]
                boxes = yolo2_utils.nms(boxes, nms_thresh)
                new_boxes = boxes[:, [6, 0, 1, 2, 3]]
                new_boxes = new_boxes[new_boxes[:, 0] == target_label]
                new_boxes = new_boxes.detach().cpu().numpy()
                if lab_dir is not None:
                    save_dir = os.path.join(lab_dir, labs[i])
                    np.savetxt(save_dir, new_boxes, fmt='%f')
                    img = unloader(data[i].detach().cpu())
                if img_dir is not None:
                    save_dir = os.path.join(img_dir, labs[i].replace('.txt', '.png'))
                    img.save(save_dir)
    print('preparing done')

def test_model(attacker, loader, target_label, conf_thresh=0.5, nms_thresh=0.4, iou_thresh=0.5, num_of_samples=100,
                old_fasion=True):
    """测试模型性能"""
    attacker.model.eval()
    total = 0.0
    proposals = 0.0
    correct = 0.0
    batch_num = len(loader)

    with torch.no_grad():
        positives = []
        for batch_idx, (data, target, img_paths, lab_paths) in tqdm(enumerate(loader), total=batch_num, position=0):
            data = data.to(attacker.device)
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
                   target_label, 
                   save_dir='./test_results', 
                   do_prepare_data=False):

    from adversarial_attacks.physical.tcega.cfg import get_cfgs
    args, kwargs = get_cfgs('yolov2', method)
    kwargs['max_lab'] = 100
    tcega = TCEGA(method=method, args=args, kwargs=kwargs)

    """运行完整的评估流程"""
    if not hasattr(tcega, 'test_cloth'):
        raise ValueError("Please load pretrained attack model first using load_pretrained_attack()")
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, f'yolov2_{method}')

    if do_prepare_data:
        prepare_data(tcega, img_ori_dir, target_label)
    
    # 创建测试数据加载器
    img_dir_test = './data/test_padded'
    lab_dir_test = f'./data/test_lab_{tcega.kwargs["name"] if tcega.kwargs else "yolov2"}'
    test_data = load_data.InriaDataset(img_dir_test,
                                        lab_dir_test, 
                                        tcega.kwargs['max_lab'], 
                                        tcega.args.img_size, 
                                        shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, tcega.args.batch_size, shuffle=False, num_workers=10)
    
    # 运行测试
    plt.figure(figsize=[15, 10])
    prec, rec, ap, confs = test_model(tcega, test_loader, target_label, conf_thresh=0.01, 
                                            old_fasion=tcega.kwargs['old_fasion'] if tcega.kwargs else True)
    
    # 保存结果
    np.savez(save_path, prec=prec, rec=rec, ap=ap, confs=confs, 
            adv_patch=tcega.cloth.detach().cpu().numpy())
    print(f'AP is {ap:.4f}')
    
    plt.plot(rec, prec)
    leg = [f'{method}: ap {ap:.3f}']
    unloader(tcega.cloth[0]).save(save_path + '.png')
    
    return prec, rec, ap, confs

if __name__ == "__main__":
    # test_basic_function()

    # img_ori_dir = './data/INRIAPerson/Test/pos'
    # target_label = 0
    # method = "TCA"

    # img_ori_dir = './dataset/coco2017_person/sub100/images/val2017'
    # target_label = 0
    # method = "TCA"

    img_ori_dir = './dataset/coco2017_car/sub100/images/val2017'
    target_label = 2
    method = "TCA"

    set_random_seed()
    run_evaluation(method=method, img_ori_dir=img_ori_dir, target_label=target_label,
                    save_dir=f'./test_results_reproduce/{method}', 
                    do_prepare_data=True)

    # 评测四种方法
    # for method_idx, method in enumerate(['RCA', 'TCA', 'EGA', 'TCEGA']):
    #     set_random_seed()
    #     preproduce_benchmark(method, img_ori_dir, prepare_data=False)