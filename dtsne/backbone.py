"""
@Time    : 2020/10/27 16:01
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : backbone.py
@Software: PyCharm
@Desc    : 
"""

import torch.nn as nn


class TeacherNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_class):
        super(TeacherNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_class = num_class
        self.freezed = False

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(negative_slope=2.5e-1, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, num_class)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.freezed = True

    def forward(self, x):
        out = self.encoder(x)

        if not self.freezed:
            out = self.classifier(out)

        return out


class StudentNet(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(StudentNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(negative_slope=2e-1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.encoder(x)

        return out
