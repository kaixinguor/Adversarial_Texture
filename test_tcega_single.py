#!/usr/bin/env python3
"""
测试TCEGA类的单图推理模式
"""

import sys
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import transforms
unloader = transforms.ToPILImage()
# 添加当前目录到Python路径
sys.path.append('.')
from tqdm import tqdm

from adversarial_attacks.physical import TCEGA
from adversarial_attacks.utils.vis_tools import set_chinese_font

set_chinese_font()

def test_single_image_inference(method, test_image_path, target_label, save_dir=None):
    """测试单图推理模式"""
    
    print("初始化TCEGA模型...")
    
    # 初始化TCEGA模型
    tcega = TCEGA(method=method,model_name='yolov2')
    
    # 加载测试图片
    test_image = Image.open(test_image_path).convert('RGB')
  
    # 预处理check
    preprocessed_image = tcega.preprocess_image(test_image)
    preprocessed_image.save("preprocessed_image.png")
    print("preprocessed_image.size: ", preprocessed_image.size)
    
    # 1. 原始图片检测
    print("\n1. 原始图片检测...")
    original_results = tcega.detect(test_image)
    print(f"   ✓ 原始图片检测完成，检测到 {len(original_results['bboxes'])} 个目标")

    # 打印检测结果详情
    if len(original_results['bboxes']) > 0:
        print("   检测结果详情:")
        for i, (bbox, score, label) in enumerate(zip(original_results['bboxes'], 
                                                        original_results['scores'], 
                                                        original_results['labels'])):
            print(f"   目标 {i+1}: 置信度={score[0]:.3f}, 类别={int(label[0])}, "
                    f"边界框=({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
    
    # from adversarial_attacks.detectors.yolo2.utils import plot_boxes_cv2
    # plot_boxes_cv2(np.array(preprocessed_image), debug_target, savename='original_results.png')

    # 2. 对抗样本生成和检测
    print("\n2. 对抗样本生成和检测...")
    try:
        adversarial_image = tcega.generate_adversarial_example(test_image, target_label)
        print("adversarial_image.size: ", adversarial_image.size)
        
        # 对对抗样本进行检测
        adversarial_results = tcega.detect(adversarial_image)
        print(f"   ✓ 对抗样本检测完成，检测到 {len(adversarial_results['bboxes'])} 个目标")
        
        # 打印对抗样本检测结果详情
        if len(adversarial_results['bboxes']) > 0:
            print("   对抗样本检测结果详情:")
            for i, (bbox, score, label) in enumerate(zip(adversarial_results['bboxes'], 
                                                        adversarial_results['scores'], 
                                                        adversarial_results['labels'])):
                print(f"   目标 {i+1}: 置信度={score[0]:.3f}, 类别={int(label[0])}, "
                      f"边界框=({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
        
        print("   ✓ 对抗样本推理成功")
        
    except Exception as e:
        print(f"   ✗ 对抗样本推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 生成对比可视化
    print("\n3. 生成对比可视化...")
    img_name = os.path.basename(test_image_path)
    try:
        create_comparison_visualization(
            test_image, 
            preprocessed_image, 
            adversarial_image, 
            original_results, 
            adversarial_results,
            tcega.class_names,
            save_path=os.path.join(save_dir,img_name)
        )
        print("   ✓ 对比可视化已保存")
    except Exception as e:
        print(f"   ✗ 可视化生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    return True

def create_comparison_visualization(original_image, processed_image, adversarial_image, 
                                  original_detections, adversarial_detections, class_names, save_path=None):
    """创建对比可视化"""
    
    # 创建大图
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 原始图片
    plt.subplot(2, 3, 1)
    plt.imshow(original_image)
    plt.title('原始输入图片', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 2. 预处理后的图片
    plt.subplot(2, 3, 2)
    plt.imshow(processed_image)
    plt.title('预处理后图片', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 3. 对抗样本
    plt.subplot(2, 3, 3)
    plt.imshow(adversarial_image)
    plt.title('对抗样本', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 4. 原始图片检测结果
    plt.subplot(2, 3, 4)
    orig_img_with_boxes = np.array(processed_image)
    if len(original_detections['bboxes']) > 0:
        bboxes = original_detections['bboxes']
        scores = original_detections['scores']
        labels = original_detections['labels']
        for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
            # TCEGA的detect方法已经返回角点格式 [x1, y1, x2, y2]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # 确保坐标在图片范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_img_with_boxes.shape[1], x2), min(orig_img_with_boxes.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                # 使用matplotlib绘制边界框，而不是直接修改像素
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='green', facecolor='none')
                plt.gca().add_patch(rect)
                
                # 获取类别名称
                class_id = int(label[0])
                class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
                
                # 添加类别名称和置信度标签
                label_text = f'{class_name}: {score[0]:.3f}'
                plt.text(x1, y1-5, label_text, 
                        color='green', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.imshow(orig_img_with_boxes)
    plt.title(f'原始图片检测结果 ({len(original_detections["bboxes"])} 个目标)', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 5. 对抗样本检测结果
    plt.subplot(2, 3, 5)
    adv_img_with_boxes = np.array(adversarial_image)
    if len(adversarial_detections['bboxes']) > 0:
        bboxes = adversarial_detections['bboxes']
        scores = adversarial_detections['scores']
        labels = adversarial_detections['labels']
        for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
            # TCEGA的detect方法已经返回角点格式 [x1, y1, x2, y2]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # 确保坐标在图片范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(adv_img_with_boxes.shape[1], x2), min(adv_img_with_boxes.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                # 使用matplotlib绘制边界框，而不是直接修改像素
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
                plt.gca().add_patch(rect)
                
                # 获取类别名称
                class_id = int(label[0])
                class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
                
                # 添加类别名称和置信度标签
                label_text = f'{class_name}: {score[0]:.3f}'
                plt.text(x1, y1-5, label_text, 
                        color='red', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.imshow(adv_img_with_boxes)
    plt.title(f'对抗样本检测结果 ({len(adversarial_detections["bboxes"])} 个目标)', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 8. 攻击效果分析
    plt.subplot(2, 3, 6)
    create_attack_effect_analysis(original_detections, adversarial_detections)
    
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果保存到：{save_path}")
    else:
        plt.show()


def create_attack_effect_analysis(original_detections, adversarial_detections):
    """创建攻击效果分析"""
    orig_count = len(original_detections['bboxes'])
    adv_count = len(adversarial_detections['bboxes'])
    
    # 计算攻击效果指标
    detection_change = adv_count - orig_count
    detection_ratio = adv_count / max(orig_count, 1)
    
    orig_scores = original_detections['scores'].flatten() if len(original_detections['scores']) > 0 else [0]
    adv_scores = adversarial_detections['scores'].flatten() if len(adversarial_detections['scores']) > 0 else [0]
    
    avg_conf_change = np.mean(adv_scores) - np.mean(orig_scores)
    
    # 创建文本显示
    analysis_text = f"""攻击效果分析:
    
目标数量变化: {detection_change:+d}
检测比例: {detection_ratio:.2f}
平均置信度变化: {avg_conf_change:+.3f}

攻击效果评估:
"""
    
    if detection_change < 0:
        analysis_text += "✓ 成功减少检测目标"
    elif detection_change > 0:
        analysis_text += "✗ 检测目标增加"
    else:
        analysis_text += "○ 目标数量无变化"
    
    if avg_conf_change < 0:
        analysis_text += "\n✓ 成功降低置信度"
    else:
        analysis_text += "\n✗ 置信度未降低"
    
    plt.text(0.5, 0.5, analysis_text, ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    plt.axis('off')


def main_single():
    method = "TCA"
    # 测试图片
    test_image_path = 'dataset/coco2017_car/sub100/images/val2017/000000008762.jpg'
    target_label = 2
    
    # 测试单图推理
    save_dir = f"result_vis_{method}_{target_label}_single"
    os.makedirs(save_dir, exist_ok=True)
    success1 = test_single_image_inference(method, test_image_path, target_label, save_dir=save_dir)

def main_batch():
    target_label = 0
    method = "TCA"
    save_dir = f"result_vis_{method}_{target_label}"
    os.makedirs(save_dir, exist_ok=True)

    img_dir = 'dataset/coco2017_car/sub100/images/val2017'
    img_names = os.listdir(img_dir)
    for img_idx, img_name in enumerate(tqdm(img_names)):
        img_path = os.path.join(img_dir, img_name)
        success1 = test_single_image_inference(method, img_path, target_label, save_dir=save_dir)

if __name__ == "__main__":
    print("=" * 50)
    print("TCEGA单图推理模式测试")
    print("=" * 50)

    # main_batch()

    main_single()
