"""
Author: Bill Wang
"""
import argparse
import os
import pdb
import shutil
import warnings

import torch
import numpy as np

from sklearn.model_selection import KFold

from rdp_tree import RDPTree
from util import dataLoading, aucPerformance, random_list, tic_time


def arg_parse(verbose=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', dest='data_path', type=str, default="data/apascal.csv")
    parser.add_argument('--save-path', dest='save_path', type=str, default="save_model/")
    parser.add_argument('--load-path', dest='load_path', type=str, default="save_model/")
    parser.add_argument('--log-path', dest='log_path', type=str, default="logs/log.log")
    parser.add_argument('--node-batch', dest='node_batch', type=int, default=30)
    parser.add_argument('--node-epoch', dest='node_epoch', type=int, default=200)  # epoch for a node training
    parser.add_argument('--eval-interval', dest='eval_interval', type=int, default=24)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=192)
    parser.add_argument('--out-c', dest='out_c', type=int, default=50)
    parser.add_argument('--lr', dest='LR', type=float, default=1e-1)
    parser.add_argument('--tree-depth', dest='tree_depth', type=int, default=8)
    # parser.add_argument('--forest-Tnum', dest='forest_Tnum', type=int, default=30)
    parser.add_argument('--filter-ratio', dest='filter_ratio', type=float,
                        default=0.05)  # filter those with high anomaly scores
    parser.add_argument('--dropout-r', dest='dropout_r', type=float, default=0.1)
    # parser.add_argument('--random-size', dest='random_size', type=int,
    #                     default=10000)  # randomly choose 1024 size of data for training
    parser.add_argument('--num-fold', dest='num_fold', type=int, default=10)
    parser.add_argument('--use-pairwise', dest='use_pairwise', action='store_true')
    parser.add_argument('--use-momentum', dest='use_momentum', action='store_true')
    parser.add_argument('--criterion', dest='criterion', type=str, default='distance',
                        choices=['distance', 'lof', 'iforest'])
    parser.add_argument('--method', dest='testing_method', type=str, default='last_layer',
                        choices=['last_layer', 'first_layer', 'level'])

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


def train():
    pass


def test():
    pass


class ClassBalancedSplit:
    def __init__(self, n_split):
        self.n_split = n_split

    def split(self, X, y):
        assert X.shape[0] == y.shape[0]
        if len(y[y==0]) < self.n_split:
            raise ValueError(f'The instance number of class {0} is not enough!')
        if len(y[y==1]) < self.n_split:
            raise ValueError(f'The instance number of class {1} is not enough!')

        class_0_inds = np.arange(len(y))[y == 0]
        class_1_inds = np.arange(len(y))[y == 1]

        np.random.shuffle(class_0_inds)
        np.random.shuffle(class_1_inds)

        seg_0_len = len(class_0_inds) // self.n_split
        seg_1_len = len(class_1_inds) // self.n_split

        for i in range(self.n_split):
            yield np.concatenate([class_0_inds[:seg_0_len], class_1_inds[:seg_1_len]]), np.concatenate([class_0_inds[seg_0_len:], class_1_inds[seg_1_len:]])
            class_0_inds = np.roll(class_0_inds, shift=seg_0_len)
            class_1_inds = np.roll(class_1_inds, shift=seg_1_len)


if __name__ == "__main__":
    USE_GPU = torch.cuda.is_available()

    args = arg_parse()
    logfile = open(args.log_path, 'w')

    x_ori, labels_ori = dataLoading(args.data_path, logfile)
    labels_ori = labels_ori.values.reshape(-1, 1)
    data_size = labels_ori.size

    rocs = []
    aps = []
    kf = ClassBalancedSplit(n_split=args.num_fold)
    for i, (train_index, test_index) in enumerate(kf.split(x_ori, labels_ori.reshape(-1))):
        if not os.path.exists(args.save_path):
            warnings.warn(f'The path {args.save_path} does not exist, created.')
            os.makedirs(args.save_path)
        else:
            warnings.warn(f'The path {args.save_path} already existed, deleted.')
            shutil.rmtree(args.save_path)
            os.mkdir(args.save_path)

        x_train, x_test = x_ori[train_index], x_ori[test_index]
        y_train, y_test = labels_ori[train_index], labels_ori[test_index]

        rdp = RDPTree(t_id=i + 1, tree_depth=args.tree_depth, filter_ratio=args.filter_ratio)

        rdp.training_process(
            x=x_train,
            labels=y_train,
            batch_size=args.batch_size,
            node_batch=args.node_batch,
            node_epoch=args.node_epoch,
            eval_interval=args.eval_interval,
            out_c=args.out_c,
            USE_GPU=USE_GPU,
            LR=args.LR,
            save_path=args.save_path,
            logfile=logfile,
            dropout_r=args.dropout_r,
            svm_flag=False,
            use_pairwise=args.use_pairwise,
            use_momentum=args.use_momentum,
            criterion=args.criterion
        )

        x_level, first_level_scores = rdp.testing_process(
            x=x_test,
            out_c=args.out_c,
            USE_GPU=USE_GPU,
            load_path=args.load_path,
            dropout_r=args.dropout_r,
            testing_method=args.testing_method,
            svm_flag=False,
            criterion=args.criterion
        )

        if args.testing_method == 'level':
            scores = x_level
        else:
            scores = first_level_scores

        roc_auc, ap = aucPerformance(scores, y_test)
        rocs.append(roc_auc)
        aps.append(ap)

    print('ROC-AUC:')
    for i, roc in enumerate(rocs):
        print(i, '.', roc)
    print('average: ', np.mean(rocs))

    print('PR-AUC:')
    for i, pr in enumerate(aps):
        print(i, '.', pr)
    print('average: ', np.mean(aps))
