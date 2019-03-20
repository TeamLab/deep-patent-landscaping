import pandas as pd
import os
import random
import argparse
import numpy as np
import string
import json
from keras.utils import to_categorical
from keras.preprocessing import sequence
from gensim.models.word2vec import Word2Vec
from tokenizer import *
from keras_transformer import get_encoders
from keras.models import Model
from keras.layers import Dense, Input, Embedding, BatchNormalization, ELU
from keras.layers.core import Dropout
from keras_metrics import precision, recall, f1score
from keras import optimizers
from transformer import PooledOutput
import pdb
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

sequence_len = 128
batch_size = 64
dropout_pct = 0.1
num_epochs = 20
random_seed = 777


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='data folder path')
    parser.add_argument('--model_path', help='model folder path')
    parser.add_argument('--data', help='data name')
    args = parser.parse_args()

    return args


def create_word_vocab(wv):
    word2idx = {}
    idx2word = {}
    for index, word in enumerate(wv.index2word):
        word2idx[word] = index
        idx2word[index] = word

    word2idx['UNK'] = len(word2idx)
    idx2word[word2idx['UNK']] = 'UNK'
    word2idx['[CLS]'] = len(word2idx)
    idx2word[word2idx['[CLS]']] = '[CLS]'
    word2idx['[SEP]'] = len(word2idx)
    idx2word[word2idx['[SEP]']] = '[SEP]'
    wv.vectors = np.vstack((wv.vectors, np.zeros((3, word_vector.vector_size))))

    return word2idx, idx2word, wv


def define_model(seq_len, num_classes, wv):
    transformer_input = Input(shape=(seq_len,), name='transformer')
    transformer = Embedding(input_dim=wv.vectors.shape[0],
                            output_dim=wv.vector_size,
                            weights=[wv.vectors],
                            input_length=seq_len,
                            trainable=False)(transformer_input)
    transformer = get_encoders(6, transformer, 8, 512, None, 'relu', 0.1)
    transformer = PooledOutput()(transformer)
    transformer = BatchNormalization()(transformer)
    transformer = ELU()(transformer)
    transformer = Dense(256)(transformer)
    transformer = Dropout(dropout_pct)(transformer)
    transformer = BatchNormalization()(transformer)
    transformer = ELU()(transformer)
    output = Dense(num_classes, activation='softmax')(transformer)
    model = Model(inputs=[transformer_input], outputs=output, name='model')

    return model


def shuffle_and_split(x, y, random_seed=random_seed):
    training_data_to_shuffle = list(zip(x, y))
    random.seed(random_seed)
    random.shuffle(training_data_to_shuffle)
    train_x, labels = zip(*training_data_to_shuffle)

    train_idx = int(len(train_x) * 0.8)
    trainX = np.array(train_x[:train_idx])
    trainY = np.array(labels[:train_idx])
    testX = np.array(train_x[train_idx:])
    testY = np.array(labels[train_idx:])

    return trainX, trainY, testX, testY


if __name__ == "__main__":
    args = parse_args()
    #data = pd.read_csv(path.join(args.data_path, args.data))
    data = pd.read_csv(args.data)

    word_vector = Word2Vec.load('./models/w2v_512.model').wv
    word2idx, idx2word, word_vector = create_word_vocab(word_vector)

    abstract_text = data.abstract_text
    tokenized_text = cleaning_sentences(abstract_text, string.punctuation)
    tokenized_indexed_text = get_indexed_token(tokenized_text, word2idx)
    padded_x = sequence.pad_sequences(np.array(tokenized_indexed_text), maxlen=sequence_len,
                padding='post', truncating='post')
    labels = to_categorical(data.valid.astype(int).values).astype(int)
    trainX, trainY, testX, testY = shuffle_and_split(padded_x, labels)
    model = define_model(sequence_len, labels.shape[-1], word_vector)
    adam = optimizers.Adam(lr=0.0001, epsilon=1e-8)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', precision, recall, f1score])
    model.fit(x={'transformer': trainX},
              y=trainY,
              batch_size=batch_size,
              epochs=num_epochs,
              validation_data=({'transformer': testX}, testY),
              #verbose=0
              )

    train_score, train_acc, train_prec, train_recall, train_f1 = model.evaluate(x={'transformer': trainX}, y=trainY, batch_size=batch_size, verbose=0)
    print('Train score: {:.4f}'.format(train_score))
    print('Train accuracy: {:.4f}'.format(train_acc))
    print('Test p/r (f1): {:.2f}/{:.2f} ({:.2f})'.format(train_prec, train_recall, train_f1))
    
    score, acc, prec, recall, f1 = model.evaluate(x={'transformer': testX}, y=testY, batch_size=batch_size, verbose=0)
    print('Test score: {:.4f}'.format(score))
    print('Test accuracy: {:.4f}'.format(acc))
    print('Test p/r (f1): {:.2f}/{:.2f} ({:.2f})'.format(prec, recall, f1))
    
    results = {"file": args.data,
               "target":"transformer",
               "train_loss": "{}".format(train_score),
               "test_loss": "{}".format(score),
               "train_f1": "{}".format(train_f1),
               "test_f1": "{}".format(f1),
               "train_acc": "{}".format(train_acc),
               "test_acc": "{}".format(acc),
               "train_prec": "{}".format(train_prec),
               "test_prec": "{}".format(prec),
               "train_recall": "{}".format(train_recall),
               "test_recall": "{}".format(recall),
               }

    with open('results.json', 'a') as outfile:
        json.dump(results, outfile)
        outfile.write("\n")

    model.save(os.path.join(args.model_path, 'transformer.h5'))
