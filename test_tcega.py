#!/usr/bin/env python3
"""
TCEGA类基本功能测试
"""

import os
import sys
import torch

def test_import():
    """测试导入功能"""
    print("测试导入功能...")
    try:
        from tcega import TCEGA
        print("✓ TCEGA类导入成功")
        return True
    except ImportError as e:
        print(f"✗ TCEGA类导入失败: {e}")
        return False

def test_class_creation():
    """测试类创建"""
    print("\n测试类创建...")
    try:
        from tcega import TCEGA
        
        # 测试CPU设备创建
        tcega = TCEGA(method='TCEGA', device='cpu')
        print("✓ TCEGA实例创建成功")
        print(f"  设备: {tcega.device}")
        print(f"  方法: {tcega.method}")
        print(f"  模型类型: {type(tcega.model)}")
        return tcega
    except Exception as e:
        print(f"✗ TCEGA实例创建失败: {e}")
        return None

def test_components(tcega):
    """测试组件初始化"""
    if tcega is None:
        return False
        
    print("\n测试组件初始化...")
    try:
        # 检查基本组件
        components = [
            ('patch_applier', tcega.patch_applier),
            ('patch_transformer', tcega.patch_transformer),
            ('prob_extractor', tcega.prob_extractor),
            ('total_variation', tcega.total_variation),
            ('tps', tcega.tps),
        ]
        
        for name, component in components:
            if component is not None:
                print(f"  ✓ {name}: {type(component)}")
            else:
                print(f"  ✗ {name}: None")
                
        # 检查GAN组件（如果适用）
        if tcega.method in ['EGA', 'TCEGA']:
            if hasattr(tcega, 'gan') and tcega.gan is not None:
                print(f"  ✓ gan: {type(tcega.gan)}")
            else:
                print(f"  ✗ gan: 未初始化")
                
        return True
    except Exception as e:
        print(f"✗ 组件检查失败: {e}")
        return False

def test_methods(tcega):
    """测试基本方法"""
    if tcega is None:
        return False
        
    print("\n测试基本方法...")
    try:
        # 测试transform方法
        if hasattr(tcega, 'transform') and tcega.transform is not None:
            print("  ✓ transform方法存在")
        else:
            print("  ✗ transform方法不存在")
            
        # 测试其他方法
        methods = ['load_dataset', 'detect', 'generate_adversarial_example', 
                  'tcega_attack', 'evaluate_attack']
        
        for method_name in methods:
            if hasattr(tcega, method_name):
                print(f"  ✓ {method_name}方法存在")
            else:
                print(f"  ✗ {method_name}方法不存在")
                
        return True
    except Exception as e:
        print(f"✗ 方法检查失败: {e}")
        return False

def test_configuration():
    """测试配置参数"""
    print("\n测试配置参数...")
    try:
        from tcega import TCEGA
        
        # 测试不同方法
        methods = ['RCA', 'TCA', 'EGA', 'TCEGA']
        
        for method in methods:
            try:
                tcega = TCEGA(method=method, device='cpu')
                print(f"  ✓ {method}方法配置成功")
                tcega = None  # 释放内存
            except Exception as e:
                print(f"  ✗ {method}方法配置失败: {e}")
                
        return True
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False

def test_basic_function():
    """主测试函数"""
    print("TCEGA类基本功能测试")
    print("=" * 50)
    
    # 测试导入
    if not test_import():
        print("导入测试失败，退出测试")
        return
        
    # 测试类创建
    tcega = test_class_creation()
    
    # 测试组件
    if tcega is not None:
        test_components(tcega)
        test_methods(tcega)
        
    # 测试配置
    test_configuration()
    
    print("\n" + "=" * 50)
    print("基本功能测试完成!")
    
    # 清理
    if tcega is not None:
        del tcega
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
def test_evaluate():
    """测试评估功能"""
    print("\n测试评估功能...")
 
    from tcega import TCEGA
    
    tcega = TCEGA(method='TCEGA', device='cpu')
    
    tcega.run_evaluation(save_dir='./test_results1')
    
    


if __name__ == "__main__":
    # test_basic_function()

    test_evaluate()
