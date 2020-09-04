"""
Author: Bill Wang
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KDTree
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.random import sample_without_replacement

MAX_GRAD_NORM = 0.1  # clip gradient
LR_GAMMA = 0.1
LR_DECAY_EPOCHS = 5000
cos_activation = False

# the init method switch only controls RN
init_method = 'kaiming'
# init_method = 'rn_orthogonal'
# init_method = 'rn_uniform'
# init_method = 'rn_normal'

MAX_INT = np.iinfo(np.int32).max
MAX_FLOAT = np.finfo(np.float32).max


class MLP(nn.Module):
    def __init__(self, in_c, out_c):
        super(MLP, self).__init__()
        # architecture def
        self.layers = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.LeakyReLU(inplace=True, negative_slope=2.5e-1)
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if cos_activation:
                    stdv = 1. / math.sqrt(m.weight.size(1))
                    m.weight.data.normal_(std=stdv)
                    if m.bias is not None:
                        # m.bias.data.normal_(std=stdv)
                        m.bias.data.uniform_(0, math.pi)
                elif init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                elif init_method == 'rn_orthogonal':
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
                elif init_method == 'rn_uniform':
                    stdv = 1. / math.sqrt(m.weight.size(1))
                    m.weight.data.uniform_(-stdv, stdv)
                    if m.bias is not None:
                        m.bias.data.uniform_(-stdv, stdv)
                elif init_method == 'rn_normal':
                    stdv = 1. / math.sqrt(m.weight.size(1))
                    m.weight.data.normal_(std=stdv)
                    if m.bias is not None:
                        m.bias.data.normal_(std=stdv)
                else:
                    raise ValueError('could not find init_method %s' % init_method)

    def forward(self, x):
        return self.layers(x)


class RTargetNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(RTargetNet, self).__init__()
        # architecture def
        c = in_c
        layers = []

        for h in [out_c]:
            layers.append(nn.Linear(c, h))
            if not cos_activation and init_method != 'rn_orthogonal':
                layers.append(nn.LeakyReLU(inplace=True, negative_slope=2.5e-1))
            c = h

        self.layers = nn.Sequential(*layers)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if cos_activation:
                    stdv = 1. / math.sqrt(m.weight.size(1))
                    m.weight.data.normal_(std=stdv)
                    if m.bias is not None:
                        # m.bias.data.normal_(std=stdv)
                        m.bias.data.uniform_(0, math.pi)
                elif init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                elif init_method == 'rn_orthogonal':
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
                elif init_method == 'rn_uniform':
                    stdv = 1. / math.sqrt(m.weight.size(1))
                    m.weight.data.uniform_(-stdv, stdv)
                    if m.bias is not None:
                        m.bias.data.uniform_(-stdv, stdv)
                elif init_method == 'rn_normal':
                    stdv = 1. / math.sqrt(m.weight.size(1))
                    m.weight.data.normal_(std=stdv)
                    if m.bias is not None:
                        m.bias.data.normal_(std=stdv)
                else:
                    raise ValueError('could not find init_method %s' % init_method)

    def forward(self, x):
        x = self.layers(x)
        if cos_activation:
            x = torch.cos(x)
        return x


class RNet(nn.Module):
    def __init__(self, in_c, out_c, dropout_r):
        super(RNet, self).__init__()

        # architecture def
        # c = in_c
        # layers = []
        #
        # for h in [out_c]:
        #     layers.append(nn.Linear(c, h))
        #     # if not cos_activation:
        #     if True:
        #         layers.append(nn.LeakyReLU(negative_slope=2e-1, inplace=True))
        #     layers.append(nn.Dropout(dropout_r))
        #     c = h
        #
        # self.layers = nn.Sequential(*layers)
        #
        # # one more layer than target network for enough capacity
        # self.fc2 = nn.Linear(out_c, out_c)

        self.layers = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.LeakyReLU(negative_slope=2e-1, inplace=True),
            nn.Dropout(dropout_r),
            nn.Linear(out_c, out_c)
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # if True:
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
                # else:
                #     stdv = 1. / math.sqrt(m.weight.size(1))
                #     # m.weight.data.uniform_(-stdv, stdv)
                #     m.weight.data.normal_(std=stdv)
                #     if m.bias is not None:
                #         # m.bias.data.uniform_(-stdv, stdv)
                #         m.bias.data.normal_(std=stdv)

    def forward(self, x):
        x = self.layers(x)
        # if cos_activation:
        # if False:
        #     x = torch.cos(x)
        return x


class RDP_Model:
    def __init__(self, in_c, out_c, logfile=None, USE_GPU=False, LR=1e-4, dropout_r=0.2, moving_ratio=0.99):
        self.r_target_net = RTargetNet(in_c, out_c)
        # self.r_target_net = RNet(in_c, out_c, dropout_r)
        # self.projection = MLP(out_c, out_c)
        self.r_net = RNet(in_c, out_c, dropout_r)
        self.USE_GPU = USE_GPU
        self.LR = LR
        self.logfile = logfile
        self.moving_ratio = moving_ratio

        print(self.r_target_net)
        if self.logfile:
            self.logfile.write(str(self.r_target_net))
        print(self.r_net)
        if self.logfile:
            self.logfile.write(str(self.r_net))
        # print(self.projection)
        # if self.logfile:
        #     self.logfile.write(str(self.projection))

        if USE_GPU:
            self.r_target_net = self.r_target_net.cuda()
            self.r_net = self.r_net.cuda()
            # self.projection = self.projection.cuda()

        # define optimizer for predict network
        # self.r_net_optim = torch.optim.Adam(self.r_net.parameters(), lr=LR)
        # self.r_net_optim = torch.optim.SGD(
        #     [{'params': self.r_net.parameters()}, {'params': self.projection.parameters()}], lr=LR, momentum=0.9)
        self.r_net_optim = torch.optim.SGD(self.r_net.parameters(), lr=LR, momentum=0.9)

        self.epoch = 0

    @torch.no_grad()
    def moving_update(self):
        for online_params, target_params in zip(self.r_net.parameters(), self.r_target_net.parameters()):
            target_params.data = target_params.data * self.moving_ratio + (1 - self.moving_ratio) * online_params.data

    def init_target_parameters(self):
        for online_params, target_params in zip(self.r_net.parameters(), self.r_target_net.parameters()):
            target_params.data.copy_(online_params.data)
            target_params.requires_grad = False

    def train_model(self, x, epoch, use_pairwise=True, use_momentum=False):
        self.r_net.train()

        x = torch.FloatTensor(x)
        x_random = copy.deepcopy(x)
        np.random.shuffle(x_random)
        x_random = torch.FloatTensor(x_random)

        if self.USE_GPU:
            x = x.cuda()
            x_random = x_random.cuda()

        if epoch % LR_DECAY_EPOCHS == 0 and self.epoch != epoch:
            self.adjust_learning_rate()
            self.epoch = epoch

        r_target = self.r_target_net(x).detach()
        r_pred = self.r_net(x)
        gap_loss = torch.mean(F.mse_loss(r_pred, r_target, reduction='none'), dim=1).mean()
        loss = gap_loss

        if use_pairwise:
            r_target_random = self.r_target_net(x_random).detach()
            r_pred_random = self.r_net(x_random)

            xy = (F.normalize(r_target, p=1, dim=1) * F.normalize(r_target_random, p=1, dim=1)).sum(dim=1)
            x_y_ = (F.normalize(r_pred, p=1, dim=1) * F.normalize(r_pred_random, p=1, dim=1)).sum(dim=1)
            pair_wise_loss = F.mse_loss(xy, x_y_)

            loss = loss + pair_wise_loss

        self.r_net_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.r_net.parameters(), MAX_GRAD_NORM)
        self.r_net_optim.step()

        if use_momentum:
            self.moving_update()

        return gap_loss.data.cpu().numpy()

    def eval_model(self, x, criterion='distance'):
        self.r_net.eval()
        x_random = copy.deepcopy(x)
        np.random.shuffle(x_random)

        if criterion == 'distance':
            x = torch.FloatTensor(x)
            x_random = torch.FloatTensor(x_random)

            if self.USE_GPU:
                x = x.cuda()
                x_random = x_random.cuda()

            r_target = self.r_target_net(x)
            r_pred = self.r_net(x)
            gap_loss = torch.mean(F.mse_loss(r_pred, r_target, reduction='none'), dim=1)

            r_target_random = self.r_target_net(x_random).detach()
            r_pred_random = self.r_net(x_random)

            xy = F.normalize(r_target, p=1, dim=1) * F.normalize(r_target_random, p=1, dim=1)
            x_y_ = F.normalize(r_pred, p=1, dim=1) * F.normalize(r_pred_random, p=1, dim=1)
            pair_wise_loss = torch.mean(F.mse_loss(xy, x_y_, reduction='none'), dim=1)
            scores = gap_loss + pair_wise_loss
            return scores.data.cpu().numpy()
        elif criterion == 'lof':
            x = torch.FloatTensor(x)
            if self.USE_GPU:
                x = x.cuda()
            with torch.no_grad():
                r_pred = self.r_net(x)

            representations = r_pred.cpu().numpy()

            clf = LocalOutlierFactor(novelty=True)
            clf.fit(representations)
            scores = 1 - clf.decision_function(representations)

            return scores
        elif criterion == 'iforest':
            x = torch.FloatTensor(x)
            if self.USE_GPU:
                x = x.cuda()
            with torch.no_grad():
                r_pred = self.r_net(x)

            representations = r_pred.cpu().numpy()

            clf = IsolationForest()
            clf.fit(representations)
            scores = 1 - clf.decision_function(representations)

            return scores
        else:
            raise ValueError('Invalid criterion!')

    def eval_model_lesinn(self, x):
        self.r_net.eval()
        x = torch.FloatTensor(x)

        if self.USE_GPU:
            x = x.cuda()

        r_pred = self.r_net(x)
        scores = self.lesinn(r_pred.data.cpu().numpy())
        return scores.squeeze()

    def lesinn(self, x_train):
        rng = np.random.RandomState(42)
        ensemble_size = 50
        subsample_size = 8
        scores = np.zeros([x_train.shape[0], 1])
        # for reproductibility purpose
        seeds = rng.randint(MAX_INT, size=ensemble_size)
        for i in range(0, ensemble_size):
            rs = np.random.RandomState(seeds[i])
            #        sid = np.random.choice(x_train.shape[0], subsample_size)
            sid = sample_without_replacement(n_population=x_train.shape[0], n_samples=subsample_size, random_state=rs)
            subsample = x_train[sid]
            kdt = KDTree(subsample, metric='euclidean')
            dists, indices = kdt.query(x_train, k=1)
            scores += dists
        scores = scores / ensemble_size
        return scores

    def adjust_learning_rate(self):
        self.LR *= LR_GAMMA
        print(' * adjust C_LR == {}'.format(self.LR))
        if self.logfile:
            self.logfile.write(' * adjust C_LR == {}\n'.format(self.LR))

        for param_group in self.r_net_optim.param_groups:
            param_group['lr'] = self.LR

    def save_model(self, path):
        dict_to_save = {
            'r_net': self.r_net.state_dict(),
            'r_target_net': self.r_target_net.state_dict(),
            # 'r_net_optim': self.r_net_optim,
            # 'LR': self.LR,
        }
        torch.save(dict_to_save, path)

    def load_model(self, name):
        states = torch.load(name)
        self.r_net.load_state_dict(states['r_net'])
        self.r_target_net.load_state_dict(states['r_target_net'])
        if 'r_net_optim' in states:
            self.r_net_optim = states['r_net_optim']
        if 'LR' in states:
            self.LR = states['LR']
