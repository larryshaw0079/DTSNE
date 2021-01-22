"""
@Time    : 2020/12/22 19:11
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : main_pyod.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.copod import COPOD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP
from pyod.models.ocsvm import OCSVM
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from dtsne.dataset import load_data

ALL_ALGORITHMS = [
    'LSCP',
    'AE',
    'COPOD',
    'OCSVM',
    'LODA'
]
ALL_DATASETS = [
    'ad.csv',
    'aid362.csv',
    'apascal.csv',
    # 'backdoor.csv',
    'bank.csv',
    # 'celeba.csv',
    # 'census.csv',
    'chess.csv',
    'cmc.csv',
    # 'creditcard.csv',
    'lung.csv',
    # 'probe.csv',
    'r10.csv',
    'secom.csv',
    # 'sf.csv',
    'u2r.csv',
    # 'w7a.csv'
]


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--algo', type=str, default=None, choices=['COPOD', 'OCSVM', 'LSCP', 'LODA'])

    parser.add_argument('--num-trial', type=int, default=10)
    parser.add_argument('--train-ratio', type=float, default=0.7)

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


def run(run_id, algo, dataset, args):
    print(f'[INFO] Run {run_id} of algorithm {algo} on dataset {dataset}...')
    np.random.seed(args.seed + run_id)

    data, labels = load_data(os.path.join('data', dataset))
    shuffle_index = np.arange(len(data))
    np.random.shuffle(shuffle_index)
    data, labels = data[shuffle_index], labels[shuffle_index]

    train_x, test_x, train_y, test_y = train_test_split(data, labels,
                                                        train_size=args.train_ratio)
    while np.count_nonzero(test_y) == 0:
        warnings.warn('No anomalies contained in the test dataset, resampling...')
        train_x, test_x, train_y, test_y = train_test_split(data, labels,
                                                            train_size=args.train_ratio)

    if algo == 'COPOD':
        clf = COPOD()
    elif algo == 'OCSVM':
        clf = OCSVM()
    elif algo == 'LSCP':
        clf = LSCP([LOF(), LOF()])
    elif algo == 'LODA':
        clf = LODA()
    elif algo == 'AE':
        if len(train_x) < 200:
            batch = 16
        elif len(train_x) < 1000:
            batch = 128
        elif len(train_x) < 5000:
            batch = 256
        elif len(train_x) < 10000:
            batch = 512
        else:
            batch = 1024
        clf = AutoEncoder(hidden_neurons=[train_x.shape[1] // 2, train_x.shape[1] // 4, train_x.shape[1] // 2],
                          batch_size=batch, epochs=50, optimizer='sgd', verbose=0)
    else:
        raise ValueError

    clf.fit(train_x)
    pred_y = clf.decision_function(test_x)
    # pred_y = clf.predict_proba(test_x, method='linear')[:, 1]

    roc = roc_auc_score(test_y, pred_y)
    pr = average_precision_score(test_y, pred_y)

    return roc, pr


if __name__ == '__main__':
    args = parse_args()

    mean_roc = np.zeros((len(ALL_DATASETS), len(ALL_ALGORITHMS)))
    mean_roc = pd.DataFrame(mean_roc, index=ALL_DATASETS, columns=ALL_ALGORITHMS)

    std_roc = np.zeros((len(ALL_DATASETS), len(ALL_ALGORITHMS)))
    std_roc = pd.DataFrame(std_roc, index=ALL_DATASETS, columns=ALL_ALGORITHMS)

    mean_pr = np.zeros((len(ALL_DATASETS), len(ALL_ALGORITHMS)))
    mean_pr = pd.DataFrame(mean_pr, index=ALL_DATASETS, columns=ALL_ALGORITHMS)

    std_pr = np.zeros((len(ALL_DATASETS), len(ALL_ALGORITHMS)))
    std_pr = pd.DataFrame(std_pr, index=ALL_DATASETS, columns=ALL_ALGORITHMS)

    for dataset in ALL_DATASETS:
        for algo in ALL_ALGORITHMS:
            roc_scores = []
            pr_scores = []
            st = time.time()
            for i in range(args.num_trial):
                roc, pr = run(i, algo, dataset, args)
                roc_scores.append(roc)
                pr_scores.append(pr)
            ed = time.time()

            print('***************************************')
            print(f'Time elapsed: {ed - st} s.')
            print('***************** ROC *****************')
            print(roc_scores)
            print(f'mean: {np.mean(roc_scores)} - {np.std(roc_scores)}')
            print('***************** PR *****************')
            print(pr_scores)
            print(f'mean: {np.mean(pr_scores)} - {np.std(pr_scores)}')

            mean_roc.loc[dataset, algo] = np.mean(roc_scores)
            std_roc.loc[dataset, algo] = np.std(roc_scores)
            mean_pr.loc[dataset, algo] = np.mean(pr_scores)
            std_pr.loc[dataset, algo] = np.std(pr_scores)

    print(mean_roc)
    print(std_roc)
    print(mean_pr)
    print(std_pr)

    mean_roc.to_csv('mean_roc.csv', index=True, float_format='%.4f')
    std_roc.to_csv('std_roc.csv', index=True, float_format='%.4f')
    mean_pr.to_csv('mean_pr.csv', index=True, float_format='%.4f')
    std_pr.to_csv('std_pr.csv', index=True, float_format='%.4f')
