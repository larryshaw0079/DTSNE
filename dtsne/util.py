"""
@Time    : 2020/10/30 21:52
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : util.py
@Software: PyCharm
@Desc    : 
"""

import numpy as np


def replaced_sampling(total_size, sampling_size):
    return np.random.choice(np.arange(total_size), size=total_size if sampling_size is None else sampling_size,
                            replace=True)
