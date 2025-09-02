#!/usr/bin/env python3
"""
TCEGA类评估测试
"""
from adversarial_attacks.physical import TCEGA

def set_random_seed():

    # 关闭随机性
    import torch
    import numpy as np
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True  # 确保CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False     # 关闭自动优化（避免非确定性）
    # torch.use_deterministic_algorithms(True)   # 强制使用确定性算法（PyTorch 1.7+）

def test_evaluate(method='TCEGA',prepare_data=False):
    """测试评估功能"""
    print(f"\n评估{method}方法...")
 
    tcega = TCEGA(method=method)
    
    img_ori_dir = './data/INRIAPerson/Test/pos'
    # img_ori_dir = './data/coco2017_person/sub100/images/val2017'

    tcega.run_evaluation(img_ori_dir=img_ori_dir,
                         save_dir=f'./test_results_reproduce/{method}', 
                         prepare_data=prepare_data
                         )

if __name__ == "__main__":
    # test_basic_function()

    # 评测四种方法
    for method_idx, method in enumerate(['RCA', 'TCA', 'EGA', 'TCEGA']):
        set_random_seed()
        test_evaluate(method, prepare_data=False)