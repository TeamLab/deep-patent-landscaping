import numpy as np
import pandas as pd
import os
from itertools import permutations, chain
import pickle


class GraphData:
    def __init__(self, dataset, target_column, feature_column, label_column):
        self.dataset = dataset
        self.target_column = target_column
        self.feature_column = feature_column
        self.label_column = label_column
        self.targets = None
        self.features = None
        self.labels = None
        self.target2idx = None
        self.idx2target = None
        self.feature2idx = None
        self.idx2feature = None
        self.adjacency_matrix = None
        self.feature_matrix = None
        self.labels = None

    def preprocessing(self):
        columns = [self.target_column, self.feature_column, self.label_column]
        self.dataset = self.dataset[columns].loc[self.dataset[self.target_column].str.len() != 0]
        self.targets = self.dataset[self.target_column]
        self.features = self.dataset[self.feature_column]
        self.labels = self.dataset[self.label_column]

        duplicated, valid, garbage = get_dup_label(targets=self.targets, labels=self.labels)
        self.targets = self.targets.apply(lambda x: remove_duplicated_code(x, duplicated))

        codeset = set(chain(*self.targets))
        self.target2idx, self.idx2target = create_vocab(codeset)
        feature_set = set(self.features)
        self.feature2idx, self.idx2feature = create_vocab(feature_set)

        return duplicated, valid, garbage

    def create_matrix(self):
        duplicated, valid, garbage = self.preprocessing()
        self.adjacency_matrix = create_adjacency_matrix(self.targets, self.target2idx)
        self.feature_matrix = create_feature_matrix(self.target2idx, self.feature2idx, self.targets, self.features)
        self.labels = create_labels(self.idx2target, valid, garbage)
        return self.adjacency_matrix, self.feature_matrix, self.labels


def create_vocab(codeset):
    code2idx = {v: k for k, v in enumerate(codeset)}
    idx2code = {k: v for k, v in enumerate(codeset)}

    return code2idx, idx2code


def create_adjacency_matrix(target_data, code2idx):
    adjacency = np.zeros([len(code2idx), len(code2idx)])
    for codes in target_data:
        for row, col in list(permutations(codes, 2)):
            row_idx, col_idx = code2idx[row], code2idx[col]
            adjacency[row_idx, col_idx] = 1

    return adjacency


def create_feature_matrix(code2idx, feature2idx, targets, features):
    feature_matrix = np.zeros([len(code2idx), len(feature2idx)])
    for idx in range(len(targets)):
        for t_idx in targets[idx]:
            feature_matrix[code2idx[t_idx]][feature2idx[features[idx]]] = 1

    return feature_matrix


def get_dup_label(targets, labels):
    valid_code = set(chain(*targets[labels == True]))
    garbage_code = set(chain(*targets[labels == False]))
    duplicated_code = valid_code & garbage_code

    print("valid :", len(valid_code), "garbage :", len(garbage_code), "duplicated :", len(duplicated_code))
    return list(duplicated_code), list(valid_code), list(garbage_code)


def remove_duplicated_code(codelist, duplicated):
    cleaned_code = []
    for code in codelist:
        if code not in duplicated:
            cleaned_code.append(code)
    return cleaned_code


def create_labels(idx2code, valid_list, garbage_list):
    labels = np.zeros((len(idx2code), 2), dtype=int)
    for idx in range(len(idx2code)):
        if idx2code[idx] in valid_list:
            labels[idx, 1] = 1
        if idx2code[idx] in garbage_list:
            labels[idx, 0] = 1
    return labels

# data_path = './data/nighttime_driver_visibility.csv'
# target_col = 'uspc_code'
# feature_col = 'family_id'
# label_col = 'valid'
#
# dataset = pd.read_csv(data_path)
#
# dataset['uspc_code'] = dataset['uspc_code'].apply(lambda x: ast.literal_eval(x)).apply(lambda x: list(set(x)))
# dataset['cpc_code'] = dataset['cpc_code'].apply(lambda x: ast.literal_eval(x)).apply(lambda x: list(set(x)))
# dataset['ipc_code'] = dataset['ipc_code'].apply(lambda x: ast.literal_eval(x)).apply(lambda x: list(set(x)))
# dataset['refs'] = dataset['refs'].apply(lambda x: ast.literal_eval(x)).apply(lambda x: list(set(x)))
#
# columns = [target_col, feature_col, label_col]
# dataset = dataset[columns].loc[dataset[target_col].str.len() != 0]
# targets = dataset[target_col]
# labels = dataset[label_col]
#
# duplicated, valid_list, garbage_list = get_dup_label(targets=targets, labels=labels)
# targets = targets.apply(lambda x: remove_duplicated_code(x, duplicated))
#
# codeset = set(chain(*targets))
# code2idx, idx2code = create_vocab(codeset)
#
# features = dataset[feature_col]
# feature_set = set(features)
#
# feature2idx, idx2featrue = create_vocab(feature_set)
# feature_ids = [feature2idx[f] for f in feature_set]
#
# print("create adjacency_matrix", len(codeset))
# adjacency_matrix = create_adjacency_matrix(targets, code2idx)
# print("create feature matrix")
# feature_matrix = create_feature_matrix(dataset, code2idx, feature2idx, targets)
#
# labels = np.zeros((len(codeset), 2), dtype=int)
# for idx in range(len(code2idx)):
#     if idx2code[idx] in valid_list:
#         labels[idx, 1] = 1
#     if idx2code[idx] in garbage_list:
#         labels[idx, 0] = 1
#
# with open(os.path.join(path, '{}.pkl'.format(args.target)), 'wb') as f:
#     pickle.dump([feature_matrix, adjacency_matrix, labels, code2idx], f, protocol=4)
