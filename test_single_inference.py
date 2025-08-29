#!/usr/bin/env python3
"""
测试TCEGA类的单图推理模式
"""

import os
import sys
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import yolo2.utils
from torchvision import transforms
unloader = transforms.ToPILImage()
# 添加当前目录到Python路径
sys.path.append('.')

from tcega import TCEGA

def set_chinese_font():
    # 设置matplotlib支持中文显示
    import matplotlib
    import matplotlib.font_manager as fm
    chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name or 'SC' in f.name]
    print("可用中文字体:", chinese_fonts)

    if 'SimHei' not in chinese_fonts:
        chinese_fonts += ['SimHei']

    matplotlib.rcParams['font.sans-serif'] = chinese_fonts # 设置中文字体
    matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

    return chinese_fonts

set_chinese_font()

def test_single_image_inference():
    """测试单图推理模式"""
    
    print("初始化TCEGA模型...")
    
    # 初始化TCEGA模型
    try:
        tcega = TCEGA(model_name='yolov2', method='TCEGA')
        print("✓ TCEGA模型初始化成功")
    except Exception as e:
        print(f"✗ TCEGA模型初始化失败: {e}")
        return False
    
    # 测试图片
    test_image_path = 'data/INRIAPerson/Test/pos/crop_000001.png'
    
    # 加载测试图片
    try:
        test_image = Image.open(test_image_path).convert('RGB')
        print(f"✓ 测试图片加载成功: {test_image.size}")
    except Exception as e:
        print(f"✗ 测试图片加载失败: {e}")
        return False
    
    # 1. 原始图片检测
    print("\n1. 原始图片检测...")
    try:
        test_image_tensor, original_results = tcega.detect(test_image)
        processed_image = unloader(test_image_tensor[0].detach().cpu())
        print(f"   ✓ 原始图片检测完成，检测到 {len(original_results['bboxes'])} 个目标")
        
        # 打印检测结果详情
        if len(original_results['bboxes']) > 0:
            print("   检测结果详情:")
            for i, (bbox, score, label) in enumerate(zip(original_results['bboxes'], 
                                                         original_results['scores'], 
                                                         original_results['labels'])):
                print(f"   目标 {i+1}: 置信度={score[0]:.3f}, 类别={int(label[0])}, "
                      f"边界框=({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
        
    except Exception as e:
        print(f"   ✗ 原始图片检测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. 对抗样本生成和检测
    print("\n2. 对抗样本生成和检测...")
    try:
        adversarial_image_tensor = tcega.generate_adversarial_example(test_image)
        adversarial_image = unloader(adversarial_image_tensor[0].detach().cpu())
        
        # 对对抗样本进行检测
        adv_img_tensor, adversarial_results = tcega.detect(adversarial_image)
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
    try:
        create_comparison_visualization(
            test_image, 
            processed_image, 
            adversarial_image, 
            original_results, 
            adversarial_results
        )
        print("   ✓ 对比可视化已保存")
    except Exception as e:
        print(f"   ✗ 可视化生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    return True

def create_comparison_visualization(original_image, processed_image, adversarial_image, 
                                  original_detections, adversarial_detections):
    """创建对比可视化"""
    
    # 创建大图
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 原始图片
    plt.subplot(3, 3, 1)
    plt.imshow(original_image)
    plt.title('原始输入图片', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 2. 预处理后的图片
    plt.subplot(3, 3, 2)
    plt.imshow(processed_image)
    plt.title('预处理后图片', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 3. 对抗样本
    plt.subplot(3, 3, 3)
    plt.imshow(adversarial_image)
    plt.title('对抗样本', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 4. 原始图片检测结果
    plt.subplot(3, 3, 4)
    orig_img_with_boxes = np.array(processed_image)
    if len(original_detections['bboxes']) > 0:
        bboxes = original_detections['bboxes']
        scores = original_detections['scores']
        labels = original_detections['labels']
        for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # 确保坐标在图片范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_img_with_boxes.shape[1], x2), min(orig_img_with_boxes.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                # 绘制边界框
                orig_img_with_boxes[y1:y2, x1:x1+3] = [0, 255, 0]  # 左边（绿色）
                orig_img_with_boxes[y1:y2, x2-3:x2] = [0, 255, 0]  # 右边（绿色）
                orig_img_with_boxes[y1:y1+3, x1:x2] = [0, 255, 0]  # 上边（绿色）
                orig_img_with_boxes[y2-3:y2, x1:x2] = [0, 255, 0]  # 下边（绿色）
                
                # 添加置信度标签
                plt.text(x1, y1-5, f'{score[0]:.3f}', 
                        color='green', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.imshow(orig_img_with_boxes)
    plt.title(f'原始图片检测结果 ({len(original_detections["bboxes"])} 个目标)', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 5. 对抗样本检测结果
    plt.subplot(3, 3, 5)
    adv_img_with_boxes = np.array(adversarial_image)
    if len(adversarial_detections['bboxes']) > 0:
        bboxes = adversarial_detections['bboxes']
        scores = adversarial_detections['scores']
        labels = adversarial_detections['labels']
        for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # 确保坐标在图片范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(adv_img_with_boxes.shape[1], x2), min(adv_img_with_boxes.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                # 绘制边界框
                adv_img_with_boxes[y1:y2, x1:x1+3] = [255, 0, 0]  # 左边（红色）
                adv_img_with_boxes[y1:y2, x2-3:x2] = [255, 0, 0]  # 右边（红色）
                adv_img_with_boxes[y1:y1+3, x1:x2] = [255, 0, 0]  # 上边（红色）
                adv_img_with_boxes[y2-3:y2, x1:x2] = [255, 0, 0]  # 下边（红色）
                
                # 添加置信度标签
                plt.text(x1, y1-5, f'{score[0]:.3f}', 
                        color='red', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.imshow(adv_img_with_boxes)
    plt.title(f'对抗样本检测结果 ({len(adversarial_detections["bboxes"])} 个目标)', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 6. 检测结果对比统计
    plt.subplot(3, 3, 6)
    create_detection_comparison_chart(original_detections, adversarial_detections)
    
    # 7. 置信度分布对比
    plt.subplot(3, 3, 7)
    create_confidence_comparison(original_detections, adversarial_detections)
    
    # 8. 攻击效果分析
    plt.subplot(3, 3, 8)
    create_attack_effect_analysis(original_detections, adversarial_detections)
    
    # 9. 保存图片
    plt.subplot(3, 3, 9)
    plt.text(0.5, 0.5, '检测结果对比\n\n原始图片: 绿色框\n对抗样本: 红色框\n\n攻击效果:\n- 目标数量变化\n- 置信度变化\n- 检测准确性', 
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_detection_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detection_comparison_chart(original_detections, adversarial_detections):
    """创建检测结果对比图表"""
    categories = ['检测目标数', '平均置信度', '最高置信度']
    
    # 计算统计数据
    orig_count = len(original_detections['bboxes'])
    adv_count = len(adversarial_detections['bboxes'])
    
    orig_scores = original_detections['scores'].flatten() if len(original_detections['scores']) > 0 else [0]
    adv_scores = adversarial_detections['scores'].flatten() if len(adversarial_detections['scores']) > 0 else [0]
    
    orig_avg_conf = np.mean(orig_scores)
    adv_avg_conf = np.mean(adv_scores)
    
    orig_max_conf = np.max(orig_scores)
    adv_max_conf = np.max(adv_scores)
    
    original_values = [orig_count, orig_avg_conf, orig_max_conf]
    adversarial_values = [adv_count, adv_avg_conf, adv_max_conf]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, original_values, width, label='原始图片', color='green', alpha=0.7)
    plt.bar(x + width/2, adversarial_values, width, label='对抗样本', color='red', alpha=0.7)
    
    plt.xlabel('检测指标')
    plt.ylabel('数值')
    plt.title('检测结果对比')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, alpha=0.3)

def create_confidence_comparison(original_detections, adversarial_detections):
    """创建置信度分布对比"""
    orig_scores = original_detections['scores'].flatten() if len(original_detections['scores']) > 0 else []
    adv_scores = adversarial_detections['scores'].flatten() if len(adversarial_detections['scores']) > 0 else []
    
    if len(orig_scores) > 0:
        plt.hist(orig_scores, bins=10, alpha=0.7, label='原始图片', color='green', density=True)
    if len(adv_scores) > 0:
        plt.hist(adv_scores, bins=10, alpha=0.7, label='对抗样本', color='red', density=True)
    
    plt.xlabel('置信度')
    plt.ylabel('密度')
    plt.title('置信度分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

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

if __name__ == "__main__":
    print("=" * 50)
    print("TCEGA单图推理模式测试")
    print("=" * 50)
    
    # 测试单图推理
    success1 = test_single_image_inference()
    
    if success1:
        print("\n" + "=" * 50)
        print("✓ 所有测试完成！")
        print("✓ 检测结果对比图已保存为: test_detection_comparison.png")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("✗ 测试过程中出现错误")
        print("=" * 50)
