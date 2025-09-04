#!/usr/bin/env python3
"""
测试TCEGA类的单图推理模式
"""
import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
unloader = transforms.ToPILImage()
import fnmatch
from tqdm import tqdm
import shutil
# 添加当前目录到Python路径
sys.path.append('.')
from ultralytics import YOLO
import os
from pathlib import Path
import torch

from adversarial_attacks.physical import TCEGA
from adversarial_attacks.utils.vis_tools import set_chinese_font
from test_tcega_single import create_comparison_visualization
from adversarial_attacks.physical.tcega.utils import label_filter, truths_length
from adversarial_attacks.detectors.yolo2 import load_data
from adversarial_attacks.detectors.yolo2 import utils as yolo2_utils
set_chinese_font()

def prepare_data(attacker, img_ori_dir, lbl_ori_dir, img_padded_dir, lbl_padded_dir):

    print("remove old data and create empty dirs")
    if os.path.exists(img_padded_dir):
        shutil.rmtree(img_padded_dir)
    if os.path.exists(lbl_padded_dir):
        shutil.rmtree(lbl_padded_dir)
    os.makedirs(img_padded_dir)
    os.makedirs(lbl_padded_dir)
 
    print('preparing gt data: padded images and labels')
    print("attacker args: ", attacker.args)
    data_nl = load_data.InriaDataset(img_ori_dir, lbl_ori_dir, attacker.kwargs['max_lab'], attacker.args.img_size, shuffle=False)
    loader_nl = torch.utils.data.DataLoader(data_nl, batch_size=attacker.args.batch_size, shuffle=False, num_workers=10)
    with torch.no_grad():
        for batch_idx, (img_batch, lab_batch, img_path_batch, lab_path_batch) in tqdm(enumerate(loader_nl), total=len(loader_nl)):
            img_batch = img_batch.to(attacker.device)
            
            for i in range(img_batch.size(0)):
                boxes = lab_batch[i].detach().cpu().numpy()
                num_gts = truths_length(boxes)
                boxes = boxes[:num_gts]

                if lbl_padded_dir is not None:
                    filename = os.path.basename(lab_path_batch[i])
                    save_path = os.path.join(lbl_padded_dir, filename)
                    np.savetxt(save_path, boxes, fmt='%f')
                    
                if img_padded_dir is not None:
                    filename = os.path.basename(img_path_batch[i]).replace('.jpg', '.png')
                    save_path = os.path.join(img_padded_dir, filename)
                    img = unloader(img_batch[i].detach().cpu())
                    img.save(save_path)
                
    print('preparing done')

def yolo_inference(yaml_path, save_dir = "runs/yolo"):
    
    # 加载模型
    model = YOLO('yolo/weights/yolov5lu.pt')

    os.makedirs(save_dir, exist_ok=True)
    
    results = model.val(
        data=yaml_path,
        batch=16,
        conf=0.3,
        iou=0.6,
        imgsz=416,
        save_txt=True,  # 保存检测结果为YOLO格式的.txt文件
        save_conf=False,  # 在txt文件中保存置信度（可选）
        project=save_dir,
        name=''
    )

    det_result_txt_dir = os.path.join(results.save_dir, "labels")

    return results, det_result_txt_dir

def batch_attack(attacker, attack_target_label, img_ori_dir, lbl_ori_dir, adv_img_dir):

    print("remove old data and create empty dirs")
    if os.path.exists(adv_img_dir):
        shutil.rmtree(adv_img_dir)
    os.makedirs(adv_img_dir)
 
    print('preparing adversarial images')
    print("attacker args: ", attacker.args)
    data_nl = load_data.InriaDataset(img_ori_dir, lbl_ori_dir, attacker.kwargs['max_lab'], attacker.args.img_size, shuffle=False)
    loader_nl = torch.utils.data.DataLoader(data_nl, batch_size=attacker.args.batch_size, shuffle=False, num_workers=10)
    with torch.no_grad():
        for batch_idx, (data, target, img_path_batch, lab_path_batch) in tqdm(enumerate(loader_nl), total=len(loader_nl)):
            print
            data = data.to(attacker.device)
            target = target.to(attacker.device)

            adv_patch = attacker.generate_adv_patch()
            adv_batch_t = attacker.patch_transformer(adv_patch, target, attack_target_label, attacker.args.img_size, 
                                                do_rotate=True, rand_loc=False,
                                                pooling=attacker.args.pooling, 
                                                old_fasion=True)
            data = attacker.patch_applier(data, adv_batch_t)
            
            for i in range(data.size(0)):
                filename = os.path.basename(img_path_batch[i]).replace('.jpg', '.png')
                save_path = os.path.join(adv_img_dir, filename)
                img = unloader(data[i].detach().cpu())
                img.save(save_path)
                
    print('attack done')


def run_tcega_eval_pipeline(method, do_prepare_data=True):
    # 初始化TCEGA模型
    print("初始化TCEGA模型...")
    from adversarial_attacks.physical.tcega.cfg import get_cfgs
    args, kwargs = get_cfgs('yolov2', method)
    kwargs['max_lab'] = 100
    tcega = TCEGA(method=method,model_name='yolov2',args=args,kwargs=kwargs)

    img_ori_dir = './dataset/coco2017_car/sub100/images/val2017'
    lbl_ori_dir = './dataset/coco2017_car/sub100/labels/val2017'
    padded_img_dir = './dataset/coco2017_car/sub100_padded/images/val2017'
    padded_lbl_dir = './dataset/coco2017_car/sub100_padded/labels/val2017'
    yolo_det_save_dir = 'runs/coco2017_car/sub100_padded'
    if do_prepare_data:
        # 准备gt数据
        prepare_data(tcega, img_ori_dir, lbl_ori_dir, padded_img_dir, padded_lbl_dir)
    
        # 进行YOLO推理
        print("yolo inference on padded data")
        yaml_path = 'ultralytics/cfg/datasets/coco-car100.yaml'
        
        results, det_result_txt_dir = yolo_inference(yaml_path,yolo_det_save_dir)
    else:
        det_result_txt_dir = os.path.join(yolo_det_save_dir, "val/labels")

    # 检查检测文件是否都存在
    det_names = os.listdir(padded_lbl_dir)
    for det_name in det_names:
        det_path = os.path.join(det_result_txt_dir, det_name)
        if not os.path.exists(det_path):
            print(f"Warning: detect file {det_path} not found, create it")
            Path(det_path).touch()
         

    # 生成对抗样本
    attack_target_label = 2
    adv_img_dir = './dataset/coco2017_car/sub100_adv/images/val2017'
    batch_attack(tcega, attack_target_label, padded_img_dir, det_result_txt_dir, adv_img_dir)
    adv_lbl_dir = './dataset/coco2017_car/sub100_adv/labels/val2017'
    if os.path.exists(adv_lbl_dir):
        shutil.rmtree(adv_lbl_dir)
    shutil.copytree(padded_lbl_dir, adv_lbl_dir)

    # 进行YOLO推理
    print("yolo inference on adv data")
    adv_yaml_path = 'ultralytics/cfg/datasets/coco-car100-adv.yaml'
    save_dir = 'runs/coco2017_car/sub100_adv'
    results, det_result_txt_dir = yolo_inference(adv_yaml_path,save_dir)
    print(results.results_dict)

if __name__ == "__main__":
    from adversarial_attacks.utils.aux_tool import set_random_seed
    set_random_seed()
    
    method = "TCEGA"
    run_tcega_eval_pipeline(method, do_prepare_data=False)