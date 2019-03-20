import numpy as np
import pandas as pd


def test_mean_vector(codes, code2idx, lookup):
    code_idx = []
    for code in codes:
        code_idx.append(code2idx[code])
    if len(code_idx) == 1:
        return lookup[code_idx[0]]
    else:
        embed_vector = lookup[code_idx[0]]
        for idx in code_idx[1:]:
            embed_vector = np.append(embed_vector, lookup[idx]).reshape(-1, 128)

        return np.mean(embed_vector, axis=0)

def embed_mean_vector(codes, code2idx, lookup):
    code_idx = []
    for code in codes:
        if code in code2idx:
            code_idx.append(code2idx[code])
    if len(code_idx) == 0:
        return np.zeros(128,)#.reshape(128, )
    elif len(code_idx) == 1:
        return lookup[code_idx[0]]
    else:
        embed_vector = lookup[code_idx[0]]
        for idx in code_idx[1:]:
            embed_vector = np.append(embed_vector, lookup[idx]).reshape(-1, 128)

        return np.mean(embed_vector, axis=0)

