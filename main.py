"""
@Time    : 2020/10/27 15:55
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : main.py.py
@Software: PyCharm
@Desc    : 
"""

import argparse
import multiprocessing
import os
import warnings
from multiprocessing import Pool

# import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from dtsne.backbone import TeacherNet, StudentNet
from dtsne.dataset import load_data, BatchTransformation


# from dtsne.util import replaced_sampling


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default='./data/apascal.csv')
    parser.add_argument('--save-path', type=str, default='./cache')
    parser.add_argument('--world-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--num-students', type=int, default=8)
    parser.add_argument('--num-trans', type=int, default=32)
    parser.add_argument('--boost-iter', type=int, default=8)
    parser.add_argument('--boost-ratio', type=float, default=0.05)

    parser.add_argument('--num-trial', type=int, default=10)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--feature-dim', type=int, default=32)

    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--classify-score', action='store_true')
    parser.add_argument('--sampling-size', type=int, default=None)
    parser.add_argument('--max-grad-norm', type=float, default=None)

    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--pretrain-lr', type=float, default=1e-3)
    parser.add_argument('--pretrain-epochs', type=int, default=100)
    parser.add_argument('--train-epochs', type=int, default=200)
    parser.add_argument('--disp-interval', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.01)

    args_parsed = parser.parse_args()

    if verbose:
        message = ''
        message += '-------------------------------- Args ------------------------------\n'
        for k, v in sorted(vars(args_parsed).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-------------------------------- End ----------------------------------'
        print(message)

    return args_parsed


def pretrain(run_id, teacher_net, train_loader, device, args):
    if args.optim == 'adam':
        teacher_optim = optim.Adam(teacher_net.parameters(), lr=args.pretrain_lr)
    elif args.optim == 'sgd':
        teacher_optim = optim.SGD(teacher_net.parameters(), lr=args.pretrain_lr, momentum=args.momentum)
    else:
        raise ValueError
    transformer = BatchTransformation(input_size=train_loader.dataset[0][0].size(-1),
                                      output_size=train_loader.dataset[0][0].size(-1),
                                      batch_size=args.batch_size, num_trans=args.num_trans,
                                      bias=False, device=device)
    criterion = nn.CrossEntropyLoss()

    teacher_net.train()
    for epoch in range(args.pretrain_epochs):
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            x, y = transformer(x)
            y_hat = teacher_net(x)

            teacher_optim.zero_grad()
            loss = criterion(y_hat, y)
            loss.backward()
            teacher_optim.step()

            losses.append(loss.item())

        if (epoch + 1) % args.disp_interval == 0:
            print(
                f'[INFO] Process {run_id}. Epoch [{epoch + 1}/{args.pretrain_epochs}]: pretrain loss {np.mean(losses):.6f}')


def train(run_id, teacher_net, student_nets, train_loader, device, args):
    student_optims = []
    for i in range(args.num_students):
        if args.optim == 'adam':
            student_optims.append(optim.Adam(student_nets[i].parameters(), lr=args.lr))
        elif args.optim == 'sgd':
            student_optims.append(optim.SGD(student_nets[i].parameters(), lr=args.lr,
                                            momentum=args.momentum))
        else:
            raise ValueError

    mse = nn.MSELoss()

    teacher_net.eval()
    for i in range(args.num_students):
        student_nets[i].train()

    for epoch in range(args.train_epochs):
        direct_dis_list = []
        pairwise_dis_list = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                target = teacher_net(x)

            current_direct_dis = []
            current_pairwise_dis = []
            for i in range(args.num_students):
                # if args.replaced_sampling:
                #     with torch.no_grad():
                #         target = teacher_net(x[:, i, :].squeeze(1))
                #     out = student_nets[i](x[:, i, :].squeeze(1))
                # else:
                out = student_nets[i](x)

                student_optims[i].zero_grad()
                direct_dis = mse(out, target)
                pairwise_dis = mse(torch.einsum('ik,jk->ij',
                                                [F.normalize(out, dim=1),
                                                 F.normalize(out, dim=1)]),
                                   torch.einsum('ik,jk->ij',
                                                [F.normalize(target, dim=1),
                                                 F.normalize(target, dim=1)]))
                loss = direct_dis + pairwise_dis
                loss.backward()
                if args.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(student_nets[i].parameters(), args.max_grad_norm)
                student_optims[i].step()
                current_direct_dis.append(direct_dis.item())
                current_pairwise_dis.append(pairwise_dis.item())
            direct_dis_list.append(np.mean(current_direct_dis))
            pairwise_dis_list.append(np.mean(current_pairwise_dis))

        if (epoch + 1) % args.disp_interval == 0:
            print(
                f'[INFO] Process {run_id}. Epoch [{epoch + 1}/{args.train_epochs}]: '
                f'direct loss {np.mean(direct_dis_list):.6f}, '
                f'pairwise loss {np.mean(pairwise_dis_list):.6f}')


def evaluate(run_id, teacher_net, student_nets, test_loader, device, args, in_training=False):
    teacher_net.eval()
    for i in range(args.num_students):
        student_nets[i].eval()

    if args.classify_score:
        ce_criterion = nn.CrossEntropyLoss(reduction='none')
        classifier = teacher_net.classifier

    direct_dis_list = []
    pairwise_dis_list = []
    classification_loss_list = []
    labels = []
    for x, y in test_loader:
        x = x.to(device)
        labels.append(y.numpy())
        y = y.to(device)

        current_direct_dis = []
        current_pairwise_dis = []
        current_classification_loss = []
        with torch.no_grad():
            target = teacher_net(x)
            for i in range(args.num_students):
                # if in_training and args.replaced_sampling:
                #     target = teacher_net(x[:, i, :].squeeze(1))
                #     out = student_nets[i](x[:, i, :].squeeze(1))
                # else:
                out = student_nets[i](x)
                direct_dis = F.mse_loss(out, target, reduction='none').mean(dim=1)
                pairwise_dis = F.mse_loss(torch.einsum('ik,jk->ij',
                                                       [F.normalize(out, dim=1),
                                                        F.normalize(out, dim=1)]),
                                          torch.einsum('ik,jk->ij',
                                                       [F.normalize(target, dim=1),
                                                        F.normalize(target, dim=1)]),
                                          reduction='none').mean(dim=1)

                if args.classify_score and not in_training:
                    classify_out = classifier(out)
                    ce_loss = ce_criterion(classify_out, y)
                    current_classification_loss.append(ce_loss.cpu().numpy().reshape(-1, 1))

                # gap = direct_dis + pairwise_dis
                current_direct_dis.append(direct_dis.cpu().numpy().reshape(-1, 1))
                current_pairwise_dis.append(pairwise_dis.cpu().numpy().reshape(-1, 1))
            current_direct_dis = np.concatenate(current_direct_dis, axis=1)
            current_pairwise_dis = np.concatenate(current_pairwise_dis, axis=1)
            if args.classify_score and not in_training:
                current_classification_loss = np.concatenate(current_classification_loss, axis=1)
        direct_dis_list.append(current_direct_dis)
        pairwise_dis_list.append(current_pairwise_dis)
        if args.classify_score and not in_training:
            classification_loss_list.append(current_classification_loss)

    direct_dis_list = np.concatenate(direct_dis_list)  # (num_sample, num_student)
    pairwise_dis_list = np.concatenate(pairwise_dis_list)  # (num_sample, num_student)
    if args.classify_score and not in_training:
        classification_loss_list = np.concatenate(classification_loss_list)
    labels = np.concatenate(labels)

    if in_training:
        # if args.replaced_sampling:
        #     gaps = None
        #     gaps_std = None
        #     gaps_mean = None
        #     scores = direct_dis_list + pairwise_dis_list
        # else:
        gaps = direct_dis_list + pairwise_dis_list
        gaps_std = np.std(gaps, axis=1)
        gaps_mean = np.mean(gaps, axis=1)

        scores = gaps_mean + args.lam * gaps_std
    else:
        if args.classify_score:
            gaps = direct_dis_list + pairwise_dis_list + args.alpha * classification_loss_list
            gaps_std = np.std(gaps, axis=1)
            gaps_mean = np.mean(gaps, axis=1)

            scores = gaps_mean + args.lam * gaps_std
        else:
            gaps = direct_dis_list + pairwise_dis_list
            gaps_std = np.std(gaps, axis=1)
            gaps_mean = np.mean(gaps, axis=1)

            scores = gaps_mean + args.lam * gaps_std

    if not in_training:
        direct_dis_anomaly = pd.DataFrame(direct_dis_list[labels == 1][:20])
        direct_dis_nomality = pd.DataFrame(direct_dis_list[labels == 0][:20])
        pairwise_dis_anomaly = pd.DataFrame(pairwise_dis_list[labels == 1][:20])
        pairwise_dis_nomality = pd.DataFrame(pairwise_dis_list[labels == 0][:20])

        print(f'[INFO] Process {run_id}.')
        print(f'direct dis anomaly: \n{direct_dis_anomaly}')
        print(f'direct dis nomality: \n{direct_dis_nomality}')
        print(f'pairwise dis anomaly: \n{pairwise_dis_anomaly}')
        print(f'pairwise dis nomality: \n{pairwise_dis_nomality}')
        if args.classify_score:
            classify_loss_anomaly = pd.DataFrame(classification_loss_list[labels == 1][:20])
            classify_loss_nomality = pd.DataFrame(classification_loss_list[labels == 0][:20])
            print(f'classify loss anomaly: \n{classify_loss_anomaly}')
            print(f'classify loss nomality: \n{classify_loss_nomality}')
        print(f'gaps anomaly: \n{pd.DataFrame(gaps[labels == 1][:20])}')
        print(f'gaps nomality: \n{pd.DataFrame(gaps[labels == 0][:20])}')

    return scores, gaps_std, gaps_mean, labels


def run(run_id, args):
    np.random.seed(args.seed + run_id)
    torch.manual_seed(args.seed + run_id)

    gpu_id = int((multiprocessing.current_process().name.split('-'))[-1]) - 1
    device = torch.device(gpu_id)

    print(f'[INFO] Process {run_id}. Running with gpu {gpu_id}...')
    data, labels = load_data(args.data_path)

    shuffle_index = np.arange(len(data))
    np.random.shuffle(shuffle_index)
    data, labels = data[shuffle_index], labels[shuffle_index]

    train_x, test_x, train_y, test_y = train_test_split(data, labels,
                                                        train_size=args.train_ratio)

    teacher_net = TeacherNet(in_dim=train_x.shape[-1], out_dim=args.feature_dim, num_class=args.num_trans)
    teacher_net = teacher_net.to(device)

    if run_id == 0:
        print(teacher_net)

    if args.pretrain:
        pretrain_dataset = TensorDataset(torch.from_numpy(train_x.astype(np.float32)),
                                         torch.from_numpy(train_y.astype(np.long)))
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, pin_memory=True,
                                     shuffle=True, drop_last=True)
        pretrain(run_id, teacher_net, pretrain_loader, device, args)

    teacher_net.freeze()

    # if args.replaced_sampling:
    #     train_x_ensemble = []
    #     train_y_ensemble = []
    #     for i in range(args.num_students):
    #         sampling_idx = replaced_sampling(len(train_x), args.sampling_size)
    #         train_x_ensemble.append(np.expand_dims(train_x[sampling_idx], axis=1))
    #         train_y_ensemble.append(np.expand_dims(train_y[sampling_idx], axis=1))
    #     train_x_ensemble = np.concatenate(train_x_ensemble, axis=1)
    #     train_y_ensemble = np.concatenate(train_y_ensemble, axis=1)
    #     train_x, train_y = train_x_ensemble, train_y_ensemble
    #     print(f'[INFO] Process {run_id}. '
    #           f'Randomly sampled {args.num_students} sub-datasets with shape {train_x.shape}...')

    for boost_it in range(args.boost_iter):
        print(f'[INFO] Process {run_id}. Running boost iter {boost_it}, training size {train_x.shape}...')

        train_dataset = TensorDataset(torch.from_numpy(train_x.astype(np.float32)),
                                      torch.from_numpy(train_y.astype(np.long)))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True,
                                  shuffle=True, drop_last=True)

        student_nets = [StudentNet(in_dim=train_x.shape[-1], out_dim=args.feature_dim) for i in
                        range(args.num_students)]
        for i in range(args.num_students):
            student_nets[i] = student_nets[i].to(device)
            if run_id == 0 and i == 0 and boost_it == 0:
                print(student_nets[i])

        train(run_id, teacher_net, student_nets, train_loader, device, args)

        boost_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True,
                                  shuffle=False, drop_last=False)
        scores, _, __, ___ = evaluate(run_id, teacher_net, student_nets, boost_loader, device, args, in_training=True)
        # if args.replaced_sampling:
        #     train_x_ensemble = []
        #     train_y_ensemble = []
        #     for i in range(args.num_students):
        #         rank_idx = np.argsort(scores[:, i].reshape(-1))
        #         selected_idx = rank_idx[:int(len(rank_idx) * (1 - args.boost_ratio))]
        #         train_x_ensemble.append(train_x[selected_idx, i:i + 1])
        #         train_y_ensemble.append(train_y[selected_idx, i:i + 1])
        #     train_x_ensemble = np.concatenate(train_x_ensemble, axis=1)
        #     train_y_ensemble = np.concatenate(train_y_ensemble, axis=1)
        #     train_x, train_y = train_x_ensemble, train_y_ensemble
        # else:
        rank_idx = np.argsort(scores)
        selected_idx = rank_idx[:int(len(rank_idx) * (1 - args.boost_ratio))]
        train_x, train_y = train_x[selected_idx], train_y[selected_idx]

    test_dataset = TensorDataset(torch.from_numpy(test_x.astype(np.float32)),
                                 torch.from_numpy(test_y.astype(np.long)))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True,
                             shuffle=True, drop_last=True)

    scores, gaps_std, gaps_mean, labels = evaluate(run_id, teacher_net, student_nets, test_loader, device, args)
    # print('***************** Anomaly *****************')
    # print(gaps_mean[labels == 1])
    # print(gaps_std[labels == 1])
    # print(f'mean: {np.mean(scores[labels==1])}')
    # print('***************** Nomality *****************')
    # print(gaps_mean[labels == 0][:20])
    # print(gaps_std[labels == 0][:20])
    # print(f'mean: {np.mean(scores[labels == 0])}')

    roc = roc_auc_score(labels, scores)
    pr = average_precision_score(labels, scores)

    print(f'[INFO] Process {run_id}. ROC: {roc}, PR: {pr}, Stopped...')

    return roc, pr


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.save_path):
        warnings.warn(f'The save path {args.save_path} dost not exist, created.')
        os.makedirs(args.save_path)

    gpu_num = args.world_size
    pool = Pool(processes=gpu_num)

    results = []

    for i in range(args.num_trial):
        result = pool.apply_async(run, (i, args))
        results.append(result)
    pool.close()
    pool.join()

    roc_scores = []
    pr_scores = []
    for result in results:
        roc, pr = result.get()
        roc_scores.append(roc)
        pr_scores.append(pr)
    print('***************** ROC *****************')
    print(roc_scores)
    print(f'mean: {np.mean(roc_scores)} - {np.std(roc_scores)}')
    print('***************** PR *****************')
    print(pr_scores)
    print(f'mean: {np.mean(pr_scores)} - {np.std(pr_scores)}')
