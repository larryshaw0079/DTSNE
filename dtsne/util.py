"""
@Time    : 2020/10/30 21:52
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : util.py
@Software: PyCharm
@Desc    : 
"""

import warnings

import numpy as np
import torch


def replaced_sampling(total_size, sampling_size):
    return np.random.choice(np.arange(total_size), size=total_size if sampling_size is None else sampling_size,
                            replace=True)


def setup_seed(seed):
    warnings.warn(f'You have chosen to seed ({seed}) training. '
                  f'This will turn on the CUDNN deterministic setting, '
                  f'which can slow down your training considerably! '
                  f'You may see unexpected behavior when restarting '
                  f'from checkpoints.')

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
