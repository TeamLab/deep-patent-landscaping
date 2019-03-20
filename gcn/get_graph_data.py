from __future__ import print_function
import os
import ast
import pickle
import argparse
import time

import pandas as pd
import numpy as np
from itertools import permutations
from keras.utils import to_categorical

# from graph import *
from gcn_input_utils import *
from utils import *
import pdb

OUTPUT_DIR = 'data'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='data name')
    parser.add_argument('--target', help='uspc, cpc, ipc, refs')
    args = parser.parse_args()

    return args


def load_data(file_path, target_col):
    data = pd.read_csv(file_path)

    if target_col == 'uspc':
        target_col = 'uspc_code'
        target = data.uspc_code
    elif target_col == 'cpc':
        target_col = 'cpc_code'
        target = data.cpc_code
    elif target_col == 'ipc':
        target_col = 'ipc_code'
        target = data.ipc_code
    elif target_col == 'refs':
        target_col = 'refs'
        target = data.refs
 
    data = data.loc[data[target_col].str.len() != 0].reset_index(drop=True)
    target = data[target_col]
    #data = data.loc[target.str.len() != 0].reset_index(drop=True)
    #target = target.loc[target.str.len() != 0].reset_index(drop=True)

    return data, target


def create_vocab(codeset):
    code2idx = {v: k for k, v in enumerate(codeset)}
    idx2code = {k: v for k, v in enumerate(codeset)}

    return code2idx, idx2code


def create_adjacency_matrix(target_data, code2idx):
    # with open('eggs.csv', 'a') as csvfile:
    #     import csv
    #     wr = csv.writer(csvfile, delimiter=' ',
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for codes in target_data:
    #         wr.writerows(list(permutations(codes, 2)))
    #
    # code_permutations = []
    # for codes in target_data:
    #     code_permutations.extend(list(permutations(codes, 2)))
    #
    # adjacency = np.zeros([len(code2idx), len(code2idx)])
    # for row, col in code_permutations:
    #     row_idx, col_idx = code2idx[row], code2idx[col]
    #     adjacency[row_idx, col_idx] = 1
    #
    # code_permutations = []
    adjacency = np.zeros([len(code2idx), len(code2idx)])

    for codes in target_data:
        for row, col in list(permutations(codes, 2)):
            row_idx, col_idx = code2idx[row], code2idx[col]
            adjacency[row_idx, col_idx] = 1

    return adjacency


def create_feature_matrix(dataset, code2idx, family2idx):
    feature_matrix = np.zeros([len(code2idx), len(family2idx)])
    for idx in range(len(dataset)):
        for t_idx in target[idx]:
            feature_matrix[code2idx[t_idx]][family2idx[data.family_id[idx]]] = 1

    return feature_matrix


if __name__ == "__main__":
    args = parse_args()
    filename = args.data.split('/')[-1]

    data, target = load_data(file_path=args.data, target_col=args.target)

    path = os.path.join(OUTPUT_DIR, filename.split('.')[0])
    if not os.path.exists(path):
        os.mkdir(path)

    target = target.apply(lambda x: ast.literal_eval(x))
    series_convert_to_list(data)
    duplicated, valid_list, garbage_list = get_dup_label(data, target)
    target = pd.Series(remove_duplicated_code(target, duplicated))

    codeset = set([code for codes in target for code in codes])
    code2idx, idx2code = create_vocab(codeset)

    family_set = set([f_id for f_id in data['family_id']])
    family2idx, idx2family = create_vocab(family_set)
    family_ids = [family2idx[f] for f in family_set]
    print("create adjacency_matrix", len(codeset))
    adjacency_matrix = create_adjacency_matrix(target, code2idx)
    print("create feature matrix")
    feature_matrix = create_feature_matrix(data, code2idx, family2idx)

    labels = []
    for i in code2idx:
        if i in valid_list:
            labels.append([0, 1])
        if i in garbage_list:
            labels.append([1, 0])
    labels = np.array(labels)

    with open(os.path.join(path, '{}.pkl'.format(args.target)), 'wb') as f:
        pickle.dump([feature_matrix, adjacency_matrix, labels, code2idx], f, protocol=4)
