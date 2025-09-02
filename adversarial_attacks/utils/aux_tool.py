import torch
import numpy as np

def set_random_seed():

    # 关闭随机性
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True  # 确保CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False     # 关闭自动优化（避免非确定性）
    # torch.use_deterministic_algorithms(True)   # 强制使用确定性算法（PyTorch 1.7+）