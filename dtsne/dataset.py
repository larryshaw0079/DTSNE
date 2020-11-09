"""
@Time    : 2020/10/27 16:01
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : dataset.py
@Software: PyCharm
@Desc    : 
"""

import pandas as pd

import torch


def load_data(data_path):
    df = pd.read_csv(data_path)
    labels = df['class'].values
    x_df = df.drop(['class'], axis=1)
    x = x_df.values

    return x, labels


class BatchTransformation:
    def __init__(self, input_size, output_size, batch_size, num_trans, bias=False, device=None):
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_trans = num_trans
        self.device = device
        self.trans_mat = torch.randn(num_trans, input_size, output_size)
        self.bias = None

        if self.device is None:
            self.device = torch.device('cpu')

        self.trans_mat = self.trans_mat.to(self.device)
        if bias:
            self.bias = torch.randn(output_size)
            self.bias = self.bias.to(self.device)

    def __call__(self, x):
        out = torch.einsum('kjm,ij->ikm', self.trans_mat, x)  # x: (batch, input_size)
        if self.bias is not None:
            out = out + self.bias

        y = torch.arange(self.num_trans, dtype=torch.long).repeat(self.batch_size, 1)
        y = y.to(self.device)

        out = out.reshape(-1, out.size(2))
        out = out.contiguous()

        # out: (batch*num_trans, output_size)
        return out, y.reshape(-1)
