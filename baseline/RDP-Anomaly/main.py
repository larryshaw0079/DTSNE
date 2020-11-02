"""
Author: Bill Wang
"""

import os
import shutil
import sys

import numpy as np
import torch
from rdp_tree import RDPTree
from sklearn.model_selection import train_test_split
from util import dataLoading, aucPerformance, tic_time
from util import random_list

data_path = "data/apascal.csv"

save_path = "save_model/"
load_path = "save_model/"
log_path = "logs/log.log"
logfile = open(log_path, 'w')
node_batch = 30
node_epoch = 200  # epoch for a node training
eval_interval = 24
batch_size = 192
train_ratio = 0.7
out_c = 50
USE_GPU = True
LR = 1e-1
tree_depth = 8
forest_Tnum = 10
filter_ratio = 0.05  # filter those with high anomaly scores
dropout_r = 0.1
random_size = 10000  # randomly choose 1024 size of data for training

# count from 1
testing_methods_set = ['last_layer', 'first_layer', 'level']
testing_method = 1

if not torch.cuda.is_available():
    USE_GPU = False

# Set mode
dev_flag = True
if dev_flag:
    print("Running in DEV_MODE!")
else:
    # running on servers
    print("Running in SERVER_MODE!")
    data_path = sys.argv[1]
    save_path = sys.argv[2]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logfile = None


def run():
    global random_size

    shutil.rmtree(save_path)
    os.mkdir(save_path)

    x_ori, labels_ori = dataLoading(data_path, logfile)
    labels_ori = labels_ori.values
    train_x, test_x, train_y, test_y = train_test_split(x_ori, labels_ori,
                                                        train_size=train_ratio)
    train_size = train_y.shape[0]
    test_size = test_y.shape[0]

    # build forest
    forest = []
    for i in range(forest_Tnum):
        forest.append(RDPTree(t_id=i + 1,
                              tree_depth=tree_depth,
                              filter_ratio=filter_ratio,
                              ))

    print("Init tic time.")
    tic_time()

    sum_result = np.zeros(test_size, dtype=np.float64)

    # training process
    for i in range(forest_Tnum):

        # random sampling with replacement
        random_pos = random_list(0, train_size - 1, random_size)
        # random sampling without replacement
        # random_pos = random.sample(range(0, data_size), random_size)

        # to form x and labels
        x = train_x[random_pos]
        labels = train_y[random_pos]
        # x = train_x
        # labels = train_y

        print("train tree id:", i, "tic time.")
        tic_time()

        forest[i].training_process(
            x=x,
            labels=labels,
            batch_size=batch_size,
            node_batch=node_batch,
            node_epoch=node_epoch,
            eval_interval=eval_interval,
            out_c=out_c,
            USE_GPU=USE_GPU,
            LR=LR,
            save_path=save_path,
            logfile=logfile,
            dropout_r=dropout_r,
        )

        print("train tree id:", i, "tic time end.")
        tic_time()

        print("test tree id:", i, "tic time.")
        tic_time()

        x = test_x
        labels = test_y

        x_level, first_level_scores = forest[i].testing_process(
            x=x,
            out_c=out_c,
            USE_GPU=USE_GPU,
            load_path=load_path,
            dropout_r=dropout_r,
            testing_method=testing_methods_set[testing_method - 1],
        )

        if testing_methods_set[testing_method - 1] == 'level':
            sum_result += x_level
            current_result = x_level
        else:
            sum_result += first_level_scores
            current_result = first_level_scores

        aucPerformance(current_result, labels)

        print("test tree id:", i, "tic time.")
        tic_time()

    scores = sum_result / forest_Tnum

    print('***************** Anomaly *****************')
    print(scores[labels == 1][:30])
    print(f'mean: {np.mean(scores[labels == 1])}, std: {np.std(scores[labels == 1])}')
    print('***************** Normality *****************')
    print(scores[labels == 0][:30])
    print(f'mean: {np.mean(scores[labels == 0])}, std: {np.std(scores[labels == 0])}')

    aucPerformance(scores, labels)


# def test(test_data, test_labels):
#     x, labels = dataLoading(data_path)
#     data_size = labels.size
#
#     # build forest
#     forest = []
#     for i in range(forest_Tnum):
#         forest.append(RDPTree(t_id=i + 1,
#                               tree_depth=tree_depth,
#                               ))
#
#     sum_result = np.zeros(data_size, dtype=np.float64)
#
#     print("Init tic time.")
#     tic_time()
#
#     # testing process
#     for i in range(forest_Tnum):
#
#         print("tree id:", i, "tic time.")
#         tic_time()
#
#         x_level, first_level_scores = forest[i].testing_process(
#             x=x,
#             out_c=out_c,
#             USE_GPU=USE_GPU,
#             load_path=load_path,
#             dropout_r=dropout_r,
#             testing_method=testing_methods_set[testing_method - 1],
#         )
#
#         if testing_methods_set[testing_method - 1] == 'level':
#             sum_result += x_level
#         else:
#             sum_result += first_level_scores
#
#         print("tree id:", i, "tic time.")
#         tic_time()
#
#     scores = sum_result / forest_Tnum
#
#     print('***************** Anomaly *****************')
#     print(scores[labels == 1][:30])
#     print(f'mean: {np.mean(scores[labels == 1])}, std: {np.std(scores[labels == 1])}')
#     print('***************** Normality *****************')
#     print(scores[labels == 0][:30])
#     print(f'mean: {np.mean(scores[labels == 0])}, std: {np.std(scores[labels == 0])}')
#
#     aucPerformance(scores, labels)


if __name__ == "__main__":
    run()
