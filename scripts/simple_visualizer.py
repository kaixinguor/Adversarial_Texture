#!/usr/bin/env python3
"""
Simple YOLO annotation visualizer - 简化版本
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adversarial_attacks.utils.coco_categories80 import COCO_INSTANCE_CATEGORY_NAMES as COCO80_LABELMAP

def visualize_yolo_annotations(image_path, annotation_path=None, save_path=None, class_names=COCO80_LABELMAP):
    """
    可视化图片和YOLO格式的标注框
    
    Args:
        image_path (str): 图片路径
        annotation_path (str): YOLO标注文件路径，如果为None则只显示图片
        save_path (str): 保存路径，如果为None则不保存
        class_names (list): 类别名称列表
    """
    # 读取图片
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist!")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # 转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width = image_rgb.shape[:2]
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    # 如果有标注文件，则绘制标注框
    if annotation_path and os.path.exists(annotation_path):
        annotations = []
        
        # 读取YOLO标注文件
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append((class_id, x_center, y_center, width, height))
        
        # 生成颜色
        np.random.seed(42)
        colors = []
        for i in range(len(class_names)):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)
        
        # 绘制每个标注框
        for annotation in annotations:
            class_id, x_center, y_center, width, height = annotation
            
            # 转换为像素坐标
            x_center_pixel = x_center * img_width
            y_center_pixel = y_center * img_height
            width_pixel = width * img_width
            height_pixel = height * img_height
            
            # 计算边界框坐标
            x_min = int(x_center_pixel - width_pixel / 2)
            y_min = int(y_center_pixel - height_pixel / 2)
            x_max = int(x_center_pixel + width_pixel / 2)
            y_max = int(y_center_pixel + height_pixel / 2)
            
            # 获取类别名称和颜色
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            color = colors[class_id % len(colors)]
            
            # 创建矩形框
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor=[c/255 for c in color], facecolor='none'
            )
            ax.add_patch(rect)
            
            # 添加标签
            ax.text(x_min, y_min - 5, class_name, 
                   color=[c/255 for c in color], fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        print(f"Found {len(annotations)} annotations in {annotation_path}")
    
    ax.set_title(f"Image: {os.path.basename(image_path)}", fontsize=14)
    ax.axis('off')
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    # 显示图片
    plt.tight_layout()
    plt.show()

def visualize_batch_images(image_dir, annotation_dir, output_dir=None, max_images=5):
    """
    批量可视化图片和标注
    
    Args:
        image_dir (str): 图片目录
        annotation_dir (str): 标注目录
        output_dir (str): 输出目录
        max_images (int): 最大显示图片数量
    """
    if not os.path.exists(image_dir):
        print(f"Error: Image directory {image_dir} does not exist!")
        return
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # 限制显示数量
    image_files = image_files[:max_images]
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 批量处理
    for i, image_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_file}")
        
        image_path = os.path.join(image_dir, image_file)
        
        # 构建标注文件路径
        annotation_file = f"{os.path.splitext(image_file)[0]}.txt"
        annotation_path = os.path.join(annotation_dir, annotation_file) if annotation_dir else None
        
        # 构建输出文件路径
        save_path = None
        if output_dir:
            save_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_annotated.png")
        
        # 可视化
        visualize_yolo_annotations(image_path, annotation_path, save_path)

# 示例用法
if __name__ == "__main__":
    # 示例1: 可视化单张图片
    visualize_yolo_annotations(
        image_path="dataset/coco2017_person/sub100/images/val2017/000000005001.jpg",
        annotation_path="dataset/coco2017_person/sub100/yolo_labels/val2017/000000005001.txt",
        save_path="test_visualization.png"
    )
    
    # 示例2: 批量可视化
    # visualize_batch_images(
    #     image_dir="data/INRIAPerson/Test/pos",
    #     annotation_dir="data/INRIAPerson/Test/yolo_labels",
    #     output_dir="visualization_results",
    #     max_images=3
    # )
    
    print("YOLO annotation visualizer ready!")
    print("Usage:")
    print("1. visualize_yolo_annotations(image_path, annotation_path, save_path)")
    print("2. visualize_batch_images(image_dir, annotation_dir, output_dir)")
