from __future__ import print_function
import pandas as pd
import os
import random
from itertools import permutations
import numpy as np
import scipy.sparse as sp
import ast
import argparse

from graph import *
from gcn_input_utils import *
import time
from utils import *
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--data', help='data name')
parser.add_argument('--target', help='uspc, cpc, ipc, refs')
args = parser.parse_args()


filename = args.data.split('/')[-1]

path = os.path.join('./data', filename.split('.')[0])
if not os.path.exists(path):
    os.mkdir(path)

data = pd.read_csv(args.data)

if args.target == 'uspc':
    target = data.uspc_code
elif args.target == 'cpc':
    target = data.cpc_code
elif args.target == 'ipc':
    target = data.ipc_code
elif args.target == 'refs':
    target = data.refs

target = target.apply(lambda x : ast.literal_eval(x))
series_convert_to_list(data)
duplicated, valid_list, garbage_list = get_dup_label(data, target)
target = pd.Series(remove_duplicated_code(target, duplicated))
data = data.loc[target.str.len() != 0].reset_index(drop=True)




all_codes = set()
for codes in target:
    for code in codes:
        all_codes.add(code)

code2idx = {v:k for k,v in enumerate(all_codes)}
idx2code = {k:v for k,v in enumerate(all_codes)}


all_family = set()
for i in data.family_id:
    all_family.add(i)

family2idx = {v:k for k,v in enumerate(all_family)}
idx2family = {k:v for k,v in enumerate(all_family)}


code_permutations = []
for codes in target:
    code_permutations.extend(list(permutations(codes, 2)))

code_pair_ids = []
for pair in code_permutations:
    code_pair_ids.append([code2idx[pair[0]], code2idx[pair[1]]])

a_matrix = np.zeros([len(all_codes), len(all_codes)])
for i in code_pair_ids:
    a_matrix[i[0]][i[1]] = 1

family_ids = []
for i in all_family:
    family_ids.append(family2idx[i])
features = np.zeros([len(code2idx), len(family2idx)])

for i in range(len(data)):
    for j in target[i]:
        features[code2idx[j]][family2idx[data.family_id[i]]] = 1

labels = []
for i in code2idx:
    if i in valid_list:
        labels.append([0, 1])
    if i in garbage_list:
        labels.append([1, 0])
labels = np.array(labels)

with open(os.path.join(path, args.target+'.pkl'), 'wb') as f:
    pickle.dump([features, a_matrix, labels, code2idx], f)