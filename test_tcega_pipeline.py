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

    det_result_txt_dir = os.path.join(save_dir, "val/labels")

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

if __name__ == "__main__":
    
    img_ori_dir = './dataset/coco2017_car/sub100/images/val2017'
    lbl_ori_dir = './dataset/coco2017_car/sub100/labels/val2017'
    
    # 初始化TCEGA模型
    print("初始化TCEGA模型...")
    method = "TCA"
    from adversarial_attacks.physical.tcega.cfg import get_cfgs
    args, kwargs = get_cfgs('yolov2', method)
    kwargs['max_lab'] = 100
    tcega = TCEGA(method=method,model_name='yolov2',args=args,kwargs=kwargs)

    # 准备gt数据
    img_padded_dir = './dataset/coco2017_car/sub100_padded/images/val2017'
    lbl_padded_dir = './dataset/coco2017_car/sub100_padded/labels/val2017'
    prepare_data(tcega, img_ori_dir, lbl_ori_dir, img_padded_dir, lbl_padded_dir)
    
    # 进行YOLO推理
    yaml_path = 'ultralytics/cfg/datasets/coco-car100.yaml'
    save_dir = 'runs/coco2017_car/sub100_padded'
    results, det_result_txt_dir = yolo_inference(yaml_path,save_dir)
    print(results.results_dict)

    # 生成对抗样本
    attack_target_label = 2
    adv_img_dir = './dataset/coco2017_car/sub100_adv/images/val2017'
    batch_attack(tcega, attack_target_label, img_padded_dir, lbl_padded_dir, adv_img_dir)

    # 进行TCEGA推理
    # tcega.load_pretrained_attack()
    # tcega.test_cloth(img_padded_dir, det_result_txt_dir)

    # tcega = TCEGA(method='TCA',model_name='yolov2')
    # target_label = 2

    # img_dir = './dataset/coco2017_car/sub100/images/val2017'
    # img_names = fnmatch.filter(os.listdir(img_dir), '*.jpg')

    # save_dir_padded = './tesst_results/ori_padded'
    # save_dir_adv_padded = './tesst_results/adv_padded'
    # os.makedirs(save_dir_padded, exist_ok=True)
    # os.makedirs(save_dir_adv_padded, exist_ok=True)

    # for img_name in tqdm(img_names):

    #     if img_name != '000000315187.jpg':
    #         continue
        
    #     test_image_path = os.path.join(img_dir, img_name)
    #     test_image = Image.open(test_image_path).convert('RGB')

    #     preprocessed_image = tcega.preprocess_image(test_image)
    #     preprocessed_image.save(os.path.join(save_dir_padded, img_name))

    #     original_results = tcega.detect(preprocessed_image)
    #     print(f"   ✓ 原始图片检测完成，检测到 {len(original_results['bboxes'])} 个目标")


    #     adversarial_image = tcega.generate_adversarial_example(test_image, target_label)
    #     adversarial_image.save(os.path.join(save_dir_adv_padded, img_name))

    #     adversarial_results = tcega.detect(adversarial_image)
    #     print(f"   ✓ 对抗样本检测完成，检测到 {len(adversarial_results['bboxes'])} 个目标")

    #     create_comparison_visualization(
    #         test_image, 
    #         preprocessed_image, 
    #         adversarial_image, 
    #         original_results, 
    #         adversarial_results,
    #         tcega.class_names
    #     )
    
    

