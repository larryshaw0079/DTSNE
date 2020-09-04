"""
Author: Bill Wang
"""
import argparse
import os
import shutil
import warnings

import torch

from rdp_tree import RDPTree
from util import dataLoading, random_list, tic_time


def arg_parse(verbose=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', dest='data_path', type=str, default="data/apascal.csv")
    parser.add_argument('--save-path', dest='save_path', type=str, default="save_model/")
    parser.add_argument('--log-path', dest='log_path', type=str, default="logs/log.log")
    parser.add_argument('--node-batch', dest='node_batch', type=int, default=30)
    parser.add_argument('--node-epoch', dest='node_epoch', type=int, default=200)  # epoch for a node training
    parser.add_argument('--eval-interval', dest='eval_interval', type=int, default=24)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=192)
    parser.add_argument('--out-c', dest='out_c', type=int, default=50)
    parser.add_argument('--lr', dest='LR', type=float, default=1e-1)
    parser.add_argument('--tree-depth', dest='tree_depth', type=int, default=8)
    parser.add_argument('--forest-Tnum', dest='forest_Tnum', type=int, default=30)
    parser.add_argument('--filter-ratio', dest='filter_ratio', type=float,
                        default=0.05)  # filter those with high anomaly scores
    parser.add_argument('--dropout-r', dest='dropout_r', type=float, default=0.1)
    parser.add_argument('--random-size', dest='random_size', type=int,
                        default=10000)  # randomly choose 1024 size of data for training
    parser.add_argument('--use-pairwise', dest='use_pairwise', action='store_true')
    parser.add_argument('--use-momentum', dest='use_momentum', action='store_true')
    parser.add_argument('--criterion', dest='criterion', type=str, default='distance',
                        choices=['distance', 'lof', 'iforest'])

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

# data_path = "data/apascal.csv"
#
# save_path = "save_model/"
# log_path = "logs/log.log"
# logfile = open(log_path, 'w')
# node_batch = 30
# node_epoch = 200  # epoch for a node training
# eval_interval = 24
# batch_size = 192
# out_c = 50
# USE_GPU = True
# LR = 1e-1
# tree_depth = 8
# forest_Tnum = 30
# filter_ratio = 0.05  # filter those with high anomaly scores
# dropout_r = 0.1
# random_size = 10000  # randomly choose 1024 size of data for training

# if not torch.cuda.is_available():


# Set mode
# dev_flag = True
# if dev_flag:
#     print("Running in DEV_MODE!")
# else:
#     # running on servers
#     print("Running in SERVER_MODE!")
#     data_path = sys.argv[1]
#     save_path = sys.argv[2]

if __name__ == "__main__":
    USE_GPU = torch.cuda.is_available()

    args = arg_parse()
    global random_size
    random_size = args.random_size
    logfile = open(args.log_path, 'w')

    if not os.path.exists(args.save_path):
        warnings.warn(f'The path {args.save_path} does not exist, created.')
        os.makedirs(args.save_path)
    else:
        warnings.warn(f'The path {args.save_path} already existed, deleted.')
        shutil.rmtree(args.save_path)
        os.mkdir(args.save_path)

    svm_flag = False
    if 'svm' in args.data_path:
        svm_flag = True
        from util import get_data_from_svmlight_file
        x_ori, labels_ori = get_data_from_svmlight_file(args.data_path)
        random_size = 1024
    else:
        x_ori, labels_ori = dataLoading(args.data_path, logfile)
    data_size = labels_ori.size

    # build forest
    forest = []
    for i in range(args.forest_Tnum):
        forest.append(RDPTree(t_id=i+1,
                              tree_depth=args.tree_depth,
                              filter_ratio=args.filter_ratio,
                              ))

    print("Init tic time.")
    tic_time()

    # training process
    for i in range(args.forest_Tnum):
        # random sampling with replacement
        if random_size < 0:
            warnings.warn(f'Using full size {args.data_size} by default...')
        if random_size > data_size:
            raise ValueError(f'`random_size` {args.random_size} exceeds data size {args.data_size}!')

        random_pos = random_list(0, data_size-1, random_size)
        # random sampling without replacement
        # random_pos = random.sample(range(0, data_size), random_size)

        # to form x and labels
        x = x_ori[random_pos]
        if svm_flag:
            labels = labels_ori[random_pos]
        else:
            labels = labels_ori[random_pos].values

        print("tree id:", i, "tic time.")
        tic_time()

        forest[i].training_process(
            x=x,
            labels=labels,
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
            svm_flag=svm_flag,
            use_pairwise=args.use_pairwise,
            use_momentum=args.use_momentum,
            criterion=args.criterion
        )

        print("tree id:", i, "tic time end.")
        tic_time()
