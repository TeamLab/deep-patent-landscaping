import numpy as np
from data_utils import load_csv, convert_code_to_idx, get_text_sequence, get_code_length
from utils import load_word_embedding, load_code_embeddings, create_code_vocab
from keras.utils import to_categorical


class PatentLandscaping:
    def __init__(self, args):
        self.args = args
        self.train = []
        self.val = []
        self.test = []
        
    def load_data(self):
        train_data, val_data, test_data = load_csv(self.args.data_path, self.args.data)
        self.train_y = to_categorical(train_data.valid)
        self.val_y = to_categorical(val_data.valid)
        self.test_y = to_categorical(test_data.valid)
        
        print('Load word embedding')
        self.word2idx, idx2word, self.word_vectors = load_word_embedding(
                                                                self.args.embedding_path,
                                                                self.args.word_embedding,
                                                                self.args.data)
        
        print('Load graph embedding')
        cpc_embed_dict, ipc_embed_dict, uspc_embed_dict = load_code_embeddings(self.args.embedding_path,
                                                                                self.args.code_embedding, self.args.data)
        
        self.cpc2idx, idx2cpc, self.cpc_vectors = create_code_vocab(cpc_embed_dict)
        self.ipc2idx, idx2ipc, self.ipc_vectors = create_code_vocab(ipc_embed_dict)
        self.uspc2idx, idx2uspc, self.uspc_vectors = create_code_vocab(uspc_embed_dict)

        self.max_cpc_len, self.max_ipc_len, self.max_uspc_len = get_code_length(train_data)
        
        print('Preparing train data')
        train_cpcs, train_ipcs, train_uspcs = convert_code_to_idx(train_data, self.max_cpc_len, self.max_ipc_len, self.max_uspc_len,
                                                                    self.cpc2idx, self.ipc2idx, self.uspc2idx)
        train_abs_sequence = get_text_sequence(train_data.abstract_text, self.word2idx, self.args.max_length)

        self.train.append(train_cpcs)
        self.train.append(train_ipcs)
        self.train.append(train_uspcs)
        self.train.append(train_abs_sequence)
        
        print('Preparing validation data')
        val_cpcs, val_ipcs, val_uspcs = convert_code_to_idx(val_data, self.max_cpc_len, self.max_ipc_len, self.max_uspc_len,
                                                                self.cpc2idx, self.ipc2idx, self.uspc2idx)
        val_abs_sequence = get_text_sequence(val_data.abstract_text, self.word2idx, self.args.max_length)
        
        self.val.append(val_cpcs)
        self.val.append(val_ipcs)
        self.val.append(val_uspcs)
        self.val.append(val_abs_sequence)

        print('preparing test data')
        test_cpcs, test_ipcs, test_uspcs = convert_code_to_idx(test_data, self.max_cpc_len, self.max_ipc_len, self.max_uspc_len,
                                                                    self.cpc2idx, self.ipc2idx, self.uspc2idx)
        test_abs_sequence = get_text_sequence(test_data.abstract_text, self.word2idx, self.args.max_length)
        
        self.test.append(test_cpcs)
        self.test.append(test_ipcs)
        self.test.append(test_uspcs)
        self.test.append(test_abs_sequence)
