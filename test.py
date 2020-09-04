"""
Author: Bill Wang
"""

import argparse

import numpy as np
import torch

from rdp_tree import RDPTree
from util import dataLoading, aucPerformance, tic_time


def arg_parse(verbose=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', dest='data_path', type=str, default="data/apascal.csv")
    parser.add_argument('--load-path', dest='load_path', type=str, default="save_model/")
    parser.add_argument('--method', dest='testing_method', type=str, default='last_layer',
                        choices=['last_layer', 'first_layer', 'level'])
    parser.add_argument('--out-c', dest='out_c', type=int, default=50)
    parser.add_argument('--tree-depth', dest='tree_depth', type=int, default=8)
    parser.add_argument('--forest-Tnum', dest='forest_Tnum', type=int, default=30)
    parser.add_argument('--dropout-r', dest='dropout_r', type=float, default=0.1)
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
# load_path = "save_model/"
# out_c = 50
# USE_GPU = True
# tree_depth = 8
# forest_Tnum = 30
# dropout_r = 0.1

# count from 1

# Set mode
# dev_flag = True
# if dev_flag:
#     print("Running in DEV_MODE!")
# else:
#     # running on servers
#     print("Running in SERVER_MODE!")
#     data_path = sys.argv[1]
#     load_path = sys.argv[2]
#     tree_depth = int(sys.argv[3])
#     testing_method = int(sys.argv[4])


def main():
    args = arg_parse()

    # testing_methods_set = ['last_layer', 'first_layer', 'level']
    # testing_method = 1

    USE_GPU = torch.cuda.is_available()

    svm_flag = False
    if 'svm' in args.data_path:
        svm_flag = True
        from util import get_data_from_svmlight_file
        x, labels = get_data_from_svmlight_file(args.data_path)
    else:
        x, labels = dataLoading(args.data_path)
    data_size = labels.size

    # build forest
    forest = []
    for i in range(args.forest_Tnum):
        forest.append(RDPTree(t_id=i + 1,
                              tree_depth=args.tree_depth,
                              ))

    sum_result = np.zeros(data_size, dtype=np.float64)

    print("Init tic time.")
    tic_time()

    # testing process
    for i in range(args.forest_Tnum):

        print("tree id:", i, "tic time.")
        tic_time()

        x_level, first_level_scores = forest[i].testing_process(
            x=x,
            out_c=args.out_c,
            USE_GPU=USE_GPU,
            load_path=args.load_path,
            dropout_r=args.dropout_r,
            testing_method=args.testing_method,
            svm_flag=svm_flag,
            criterion=args.criterion
        )

        if args.testing_method == 'level':
            sum_result += x_level
        else:
            sum_result += first_level_scores

        print("tree id:", i, "tic time.")
        tic_time()

    scores = sum_result / args.forest_Tnum

    # clf = LocalOutlierFactor(novelty=True)
    # clf.fit(x)
    # scores = 1-clf.decision_function(x)

    aucPerformance(scores, labels)


if __name__ == "__main__":
    main()
