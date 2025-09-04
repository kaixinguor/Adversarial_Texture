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
# 添加当前目录到Python路径
sys.path.append('.')

from adversarial_attacks.physical import TCEGA
from adversarial_attacks.utils.vis_tools import set_chinese_font
from test_tcega_single import create_comparison_visualization
set_chinese_font()

if __name__ == "__main__":
    """测试单图推理模式"""
    
    print("初始化TCEGA模型...")
    
    # 初始化TCEGA模型
    tcega = TCEGA(method='TCA',model_name='yolov2')
    target_label = 2

    img_dir = './dataset/coco2017_car/sub100/images/val2017'
    img_names = fnmatch.filter(os.listdir(img_dir), '*.jpg')

    save_dir_padded = './tesst_results/ori_padded'
    save_dir_adv_padded = './tesst_results/adv_padded'
    os.makedirs(save_dir_padded, exist_ok=True)
    os.makedirs(save_dir_adv_padded, exist_ok=True)

    for img_name in tqdm(img_names):

        if img_name != '000000315187.jpg':
            continue
        
        test_image_path = os.path.join(img_dir, img_name)
        test_image = Image.open(test_image_path).convert('RGB')

        preprocessed_image = tcega.preprocess_image(test_image)
        preprocessed_image.save(os.path.join(save_dir_padded, img_name))

        original_results = tcega.detect(preprocessed_image)
        print(f"   ✓ 原始图片检测完成，检测到 {len(original_results['bboxes'])} 个目标")


        adversarial_image = tcega.generate_adversarial_example(test_image, target_label)
        adversarial_image.save(os.path.join(save_dir_adv_padded, img_name))

        adversarial_results = tcega.detect(adversarial_image)
        print(f"   ✓ 对抗样本检测完成，检测到 {len(adversarial_results['bboxes'])} 个目标")

        create_comparison_visualization(
            test_image, 
            preprocessed_image, 
            adversarial_image, 
            original_results, 
            adversarial_results,
            tcega.class_names
        )
    
    

