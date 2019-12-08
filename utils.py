import os
import ast
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from layers import EmbeddingZero
from keras.regularizers import l2


def create_word_vocab(wv):
    word2idx = {}
    idx2word = {}
    word2idx['[PAD]'] = 0
    idx2word[0] = '[PAD]'
    for index, word in enumerate(wv.index2word):
        word2idx[word] = index+1
        idx2word[index+1] = word
    word2idx['UNK'] = len(word2idx)
    idx2word[word2idx['UNK']] = 'UNK'
    word2idx['[CLS]'] = len(word2idx)
    idx2word[word2idx['[CLS]']] = '[CLS]'
    word2idx['[SEP]'] = len(word2idx)
    idx2word[word2idx['[SEP]']] = '[SEP]'
    wv.vectors = np.vstack((np.zeros((1, wv.vector_size)), wv.vectors))
    wv.vectors = np.vstack((wv.vectors, np.zeros((3, wv.vector_size))))
    return word2idx, idx2word, wv


def create_code_vocab(code_dict):
    code2idx = {}
    idx2code = {}
    vectors = []
    embedding_size = len(list(code_dict.values())[0])
    code2idx['[PAD]'] = 0
    idx2code[0] = '[PAD]'
    vectors.append([0]*embedding_size)
    for i, (k, v) in enumerate(code_dict.items()):
        code2idx[k] = i+1
        idx2code[i+1] = k
        vectors.append(v)
    code2idx['UNK'] = len(idx2code)
    idx2code[code2idx['UNK']] = 'UNK'
    vectors.append([0]*embedding_size)
    return code2idx, idx2code, np.array(vectors)


def load_word_embedding(embedding_path, word_embedding, data_name):
    if word_embedding == 'w2v':
        word_vectors = Word2Vec.load(os.path.join(embedding_path, word_embedding,\
                                                  data_name+'_{}.model'.format(word_embedding))).wv
        word2idx, idx2word, word_vectors = create_word_vocab(word_vectors)
        return word2idx, idx2word, word_vectors
    
    
def df_to_embed_dict(df):
    df.embedding = df.embedding.apply(lambda x : ast.literal_eval(x))
    embed_dict = {}
    for i in range(len(df)):
        node = df.iloc[i].node
        embedding = np.array(df.iloc[i].embedding)
        embed_dict[node] = embedding
    return embed_dict


def load_code_embeddings(embedding_path, code_embedding, data_name):
    cpc_embedding_df = pd.read_csv(os.path.join(embedding_path, code_embedding, data_name+'_cpc_embeddings.csv'))
    cpc_embed_dict = df_to_embed_dict(cpc_embedding_df)
    ipc_embedding_df = pd.read_csv(os.path.join(embedding_path, code_embedding, data_name+'_ipc_embeddings.csv'))
    ipc_embed_dict = df_to_embed_dict(ipc_embedding_df)
    uspc_embedding_df = pd.read_csv(os.path.join(embedding_path, code_embedding, data_name+'_uspc_embeddings.csv'))
    uspc_embed_dict = df_to_embed_dict(uspc_embedding_df)
    return cpc_embed_dict, ipc_embed_dict, uspc_embed_dict

    
def get_embedding_layer(code2idx, pretrained_embedding):
    embedding_layer = EmbeddingZero(
                        output_dim=pretrained_embedding.shape[-1],
                        input_dim=len(code2idx),
                        activity_regularizer=l2(0),
                        mask_zero=None,
                        trainable=True)
    embedding_layer.build((None,))
    embedding_layer.set_weights([pretrained_embedding])
    return embedding_layer
