import os
import re
import ast
import numpy as np
import pandas as pd
import string
from keras.preprocessing.sequence import pad_sequences


def preprocess(data):
    data.cpc_code = data.cpc_code.apply(lambda x : ast.literal_eval(x))
    data.ipc_code = data.ipc_code.apply(lambda x : ast.literal_eval(x))
    data.uspc_code = data.uspc_code.apply(lambda x : ast.literal_eval(x))
    return data


def load_csv(data_path, data_name):
    train_data = pd.read_csv(os.path.join(data_path, data_name+'_new_train.csv'))
    train_data = preprocess(train_data)
    val_data = pd.read_csv(os.path.join(data_path, data_name+'_new_val.csv'))
    val_data = preprocess(val_data)
    test_data = pd.read_csv(os.path.join(data_path, data_name+'_new_test.csv'))
    test_data = preprocess(test_data)
    return train_data ,val_data, test_data


def get_code_sequence(data, target, max_len, code_dict):
    data[target] = data[target].apply(lambda x: np.array([code_dict[i] if i in code_dict.keys() else 0 for i in x]))
    codes = pad_sequences(data[target].tolist(), maxlen=max_len, padding='post', truncating='post')
    return codes


def get_code_length(data):
    cpc_len = max(data.cpc_code.apply(lambda x: len(x)))
    ipc_len = max(data.ipc_code.apply(lambda x: len(x)))
    uspc_len = max(data.uspc_code.apply(lambda x: len(x)))
    return cpc_len, ipc_len, uspc_len


def convert_code_to_idx(data, cpc_len, ipc_len, uspc_len, cpc2idx, ipc2idx, uspc2idx):
    cpcs = get_code_sequence(data, 'cpc_code', cpc_len, cpc2idx)
    ipcs = get_code_sequence(data, 'ipc_code', ipc_len, ipc2idx)
    uspcs = get_code_sequence(data, 'uspc_code', uspc_len, uspc2idx)
    return cpcs, ipcs, uspcs


def cleaning_sentences(sentences, punctuations):
    re_punc = re.compile('[%s]' % re.escape(punctuations))
    return [[re_punc.sub("", w).lower() for w in s.split()
            if re_punc.sub("", w).lower().isalpha()] for s in sentences]


def get_indexed_sequence(text_tokens, word2idx, sequence_len):
    indexed_text_sequence = []
    for i in range(len(text_tokens)):
        text = text_tokens[i]
        word_indices = []
        for word in text:
            if word in word2idx:
                word_idx = word2idx[word]
            else:
                word_idx = word2idx['UNK']
            if len(word_indices) < sequence_len-2:
                word_indices.append(word_idx)
        word_indices = np.insert(word_indices, 0, word2idx['[CLS]'])
        word_indices = np.insert(word_indices, len(word_indices), word2idx['[SEP]'])
        indexed_text_sequence.append(word_indices)
    return indexed_text_sequence


def get_text_sequence(text, word2idx, sequence_len):
    tokenized_text = cleaning_sentences(text, string.punctuation)
    tokenized_indexed_text = get_indexed_sequence(tokenized_text, word2idx, sequence_len)
    text_sequence = pad_sequences(np.array(tokenized_indexed_text), maxlen=sequence_len,\
                                         padding='post', truncating='post')
    return text_sequence

