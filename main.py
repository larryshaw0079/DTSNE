"""
@Time    : 2020/10/27 15:55
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : main.py.py
@Software: PyCharm
@Desc    : 
"""

import argparse
import itertools
import multiprocessing
import os
import warnings
from multiprocessing import Pool

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


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default='./data/apascal.csv')
    parser.add_argument('--save-path', type=str, default='./cache')
    parser.add_argument('--only-evaluate', action='store_true')
    parser.add_argument('--world-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--num-students', type=int, default=8)
    parser.add_argument('--num-trans', type=int, default=32)
    parser.add_argument('--boost-iter', type=int, default=4)
    parser.add_argument('--boost-ratio', type=float, default=0.05)

    parser.add_argument('--num-trial', type=int, default=10)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--feature-dim', type=int, default=32)

    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--classify-score', type=str, default='none', choices=['none', 'cp', 'ne', 'ce'])
    # parser.add_argument('--score', type=str, default='dis', choices=['dis'])
    parser.add_argument('--sampling-size', type=int, default=None)
    parser.add_argument('--max-grad-norm', type=float, default=None)

    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--pretrain-lr', type=float, default=1e-3)
    parser.add_argument('--pretrain-epochs', type=int, default=100)
    parser.add_argument('--train-epochs', type=int, default=200)
    parser.add_argument('--disp-interval', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--wd', type=float, default=1e-3)
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
        teacher_optim = optim.Adam(teacher_net.parameters(), lr=args.pretrain_lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        teacher_optim = optim.SGD(teacher_net.parameters(), lr=args.pretrain_lr,
                                  weight_decay=args.wd, momentum=args.momentum)
    else:
        raise ValueError
    transformer = BatchTransformation(input_size=train_loader.dataset[0][0].size(-1),
                                      batch_size=args.batch_size, num_trans=args.num_trans,
                                      bias=False, device=device)
    criterion = nn.CrossEntropyLoss()

    teacher_net.train()
    for epoch in range(args.pretrain_epochs):
        losses = []
        for x, y in train_loader:
            x = x.to(device)

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
            student_optims.append(optim.Adam(student_nets[i].parameters(), lr=args.lr, weight_decay=args.wd))
        elif args.optim == 'sgd':
            student_optims.append(optim.SGD(student_nets[i].parameters(), lr=args.lr, weight_decay=args.wd,
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

    if args.classify_score != 'none':
        classifier = teacher_net.classifier
        transformer = BatchTransformation(input_size=test_loader.dataset[0][0].size(-1),
                                          batch_size=args.batch_size, num_trans=args.num_trans,
                                          bias=False, device=device)

    direct_dis_list = []
    pairwise_dis_list = []
    labels = []
    for x, y in test_loader:
        x = x.to(device)
        labels.append(y.numpy())

        current_direct_dis = []
        current_pairwise_dis = []
        with torch.no_grad():
            target = teacher_net(x)
            for i in range(args.num_students):
                out = student_nets[i](x)
                direct_dis = F.mse_loss(out, target, reduction='none').mean(dim=1)
                pairwise_dis = F.mse_loss(torch.einsum('ik,jk->ij',
                                                       [F.normalize(out, dim=1),
                                                        F.normalize(out, dim=1)]),
                                          torch.einsum('ik,jk->ij',
                                                       [F.normalize(target, dim=1),
                                                        F.normalize(target, dim=1)]),
                                          reduction='none').mean(dim=1)

                # gap = direct_dis + pairwise_dis
                current_direct_dis.append(direct_dis.cpu().numpy().reshape(-1, 1))
                current_pairwise_dis.append(pairwise_dis.cpu().numpy().reshape(-1, 1))

        current_direct_dis = np.concatenate(current_direct_dis, axis=1)
        current_pairwise_dis = np.concatenate(current_pairwise_dis, axis=1)

        direct_dis_list.append(current_direct_dis)
        pairwise_dis_list.append(current_pairwise_dis)

    if args.classify_score != 'none' and not in_training:
        classify_preds = []
        classify_labels = []
        for x, y in test_loader:
            x = x.to(device)
            batch_size, *_ = x.shape
            x, y = transformer(x)

            classify_out_list = []
            with torch.no_grad():
                for i in range(args.num_students):
                    out = student_nets[i](x)
                    classify_out = classifier(out)  # (batch*num_trans, num_class)
                    classify_out = classify_out.view(batch_size, args.num_trans, args.num_trans)
                    classify_out_list.append(classify_out)

            classify_out_list = torch.stack(classify_out_list, dim=1)  # (batch, num_students, num_trans, num_class)
            y = y.view(batch_size, args.num_trans)

            classify_preds.append(classify_out_list)
            classify_labels.append(y)

        classify_preds = torch.cat(classify_preds, dim=0)  # (num_sample, num_students, num_trans, num_class)
        classify_labels = torch.cat(classify_labels, dim=0)  # (num_sample, num_class)

    direct_dis_list = np.concatenate(direct_dis_list)  # (num_sample, num_student)
    pairwise_dis_list = np.concatenate(pairwise_dis_list)  # (num_sample, num_student)

    if args.classify_score != 'none' and not in_training:
        if args.classify_score == 'cp':
            classify_prob = F.softmax(classify_preds, dim=-1)  # (num_sample, num_students, num_trans, num_class)
            classify_scores = torch.zeros(*(classify_prob.size()[:-1]))
            classify_scores = classify_scores.to(device)
            for i, j, k in itertools.product(range(classify_prob.size(0)),
                                             range(classify_prob.size(1)),
                                             range(classify_prob.size(2))):
                classify_scores[i, j, k] = classify_prob[i, j, k, k]
            classify_scores = classify_scores.mean(dim=-1)
        elif args.classify_score == 'ne':
            classify_scores = F.softmax(classify_preds, dim=-1)  # (num_sample, num_students, num_trans, num_class)
            classify_scores = -classify_scores * torch.log2(classify_scores + 1e-10)
            classify_scores = classify_scores.sum(dim=-1).mean(dim=-1)
        elif args.classify_score == 'ce':
            classify_preds = classify_preds.permute(0, 3, 1, 2)  # (num_sample, num_class, num_students, num_trans)
            classify_labels = classify_labels.unsqueeze(dim=1).repeat((1, args.num_students, 1))
            classify_scores = F.cross_entropy(classify_preds, classify_labels,
                                              reduction='none')  # (num_sample, num_students, num_trans)
            classify_scores = classify_scores.mean(dim=-1)
        else:
            raise ValueError
        classify_scores = classify_scores.cpu().numpy()
        # print(f'***********{classify_scores.shape}************')
    labels = np.concatenate(labels)

    if in_training:
        gaps = direct_dis_list + pairwise_dis_list
        gaps_std = np.std(gaps, axis=1)
        gaps_mean = np.mean(gaps, axis=1)

        scores = gaps_mean + args.lam * gaps_std
    else:
        if args.classify_score != 'none':
            gaps = direct_dis_list + pairwise_dis_list + args.alpha * classify_scores
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
        if args.classify_score != 'none':
            classify_loss_anomaly = pd.DataFrame(classify_scores[labels == 1][:20])
            classify_loss_nomality = pd.DataFrame(classify_scores[labels == 0][:20])
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
    while np.count_nonzero(test_y) == 0:
        warnings.warn('No anomalies contained in the test dataset, resampling...')
        train_x, test_x, train_y, test_y = train_test_split(data, labels,
                                                            train_size=args.train_ratio)

    teacher_net = TeacherNet(in_dim=train_x.shape[-1], out_dim=args.feature_dim, num_class=args.num_trans)
    teacher_net = teacher_net.to(device)

    if run_id == 0:
        print(teacher_net)

    if args.only_evaluate:
        print(f'[INFO] Process {run_id}. Running in evaluation mode...')
        teacher_net.load_state_dict(torch.load(os.path.join(args.save_path, f'run{run_id}_teacher.pth')))
        teacher_net.freeze()

        student_nets = [StudentNet(in_dim=train_x.shape[-1], out_dim=args.feature_dim) for i in
                        range(args.num_students)]
        for i in range(args.num_students):
            student_nets[i] = student_nets[i].to(device)
            student_nets[i].load_state_dict(torch.load(os.path.join(args.save_path, f'run{run_id}_student_{i}.pth')))
            if run_id == 0 and i == 0:
                print(student_nets[i])
    else:
        if args.pretrain:
            pretrain_dataset = TensorDataset(torch.from_numpy(train_x.astype(np.float32)),
                                             torch.from_numpy(train_y.astype(np.long)))
            pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, pin_memory=True,
                                         shuffle=True, drop_last=True)
            pretrain(run_id, teacher_net, pretrain_loader, device, args)

        torch.save(teacher_net.state_dict(), os.path.join(args.save_path, f'run{run_id}_teacher.pth'))

        teacher_net.freeze()

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
            scores, _, __, ___ = evaluate(run_id, teacher_net, student_nets, boost_loader, device, args,
                                          in_training=True)

            rank_idx = np.argsort(scores)
            selected_idx = rank_idx[:int(len(rank_idx) * (1 - args.boost_ratio))]
            train_x, train_y = train_x[selected_idx], train_y[selected_idx]

        for i, student in enumerate(student_nets):
            torch.save(student.state_dict(), os.path.join(args.save_path, f'run{run_id}_student_{i}.pth'))

    test_dataset = TensorDataset(torch.from_numpy(test_x.astype(np.float32)),
                                 torch.from_numpy(test_y.astype(np.long)))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True,
                             shuffle=True, drop_last=False)

    scores, gaps_std, gaps_mean, labels = evaluate(run_id, teacher_net, student_nets, test_loader, device, args)

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
