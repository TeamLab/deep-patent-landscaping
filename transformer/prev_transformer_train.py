import pandas as pd
import os
import random
import argparse
import numpy as np


from keras.preprocessing import sequence
from gensim.models.word2vec import Word2Vec
from tokenizer import *
from keras_transformer import get_encoders
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Embedding, BatchNormalization, ELU, Concatenate, Reshape
from keras.layers.merge import concatenate
from keras.layers.core import Dropout
from keras_metrics import precision, recall, f1score
from keras.activations import softmax
from keras import optimizers
from transformer import PooledOutput


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='data folder path')
parser.add_argument('--model_path', help='model folder path')
parser.add_argument('--data', help='data name')
args = parser.parse_args()

sequence_len = 128
batch_size = 64
dropout_pct = 0.1
num_epochs = 10

data = pd.read_csv(os.path.join(args.data_path, args.data))

w2v = Word2Vec.load('./models/w2v_512.model')

word2idx = {}
idx2word = {}
index = 0
for word in w2v.wv.index2word:
    word2idx[word] = index
    idx2word[index] = word
    index += 1

word2idx['UNK'] = len(word2idx)
idx2word[word2idx['UNK']] = 'UNK'
word2idx['[CLS]'] = len(word2idx)
idx2word[word2idx['[CLS]']] = '[CLS]'
word2idx['[SEP]'] = len(word2idx)
idx2word[word2idx['[SEP]']] = '[SEP]'
# w2v.wv.vectors = np.vstack((w2v.wv.vectors, np.zeros(w2v.wv.vector_size)))

abstract_text = data.abstract_text
tokenized_text = cleaning_sentences(abstract_text, string.punctuation)
tokenized_indexed_text = get_indexed_token(tokenized_text, word2idx)

padded_x = sequence.pad_sequences(np.array(tokenized_indexed_text), maxlen=sequence_len,
            padding='post', truncating='post')

labels_indexed = []
labels = data.valid
for label in labels:
    if label:
        labels_indexed.append([0, 1])
    else:
        labels_indexed.append([1, 0])
labels_indexed = np.array(labels_indexed)

training_data_to_shuffle = list(zip(padded_x, labels_indexed))
random_seed = 777
random.seed(random_seed)
random.shuffle(training_data_to_shuffle)
train_x, labels = zip(*training_data_to_shuffle)

train_idx = int(len(train_x) * 0.8)
train_idx = int(len(train_x) * 0.8)
trainX = np.array(train_x[:train_idx])
trainY = np.array(labels[:train_idx])
testX = np.array(train_x[train_idx:])
testY = np.array(labels[train_idx:])

transformer_input = Input(shape=(trainX.shape[1],), name='transformer')
transformer = Embedding(w2v.wv.vectors.shape[0], w2v.wv.vector_size, weights=[w2v.wv.vectors], input_length=trainX.shape[1], trainable=False)(transformer_input)
transformer = get_encoders(6, transformer, 8, 512, None, 'relu', 0.1)
transformer = PooledOutput()(transformer)
transformer = BatchNormalization()(transformer)
transformer = ELU()(transformer)
transformer = Dense(256)(transformer)
transformer = Dropout(dropout_pct)(transformer)
transformer = BatchNormalization()(transformer)
transformer = ELU()(transformer)
output = Dense(2, activation='softmax')(transformer)

adam = optimizers.Adam(lr=0.0001, epsilon=1e-8)
model = Model(inputs=[transformer_input], outputs=output, name='model')
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', precision, recall, f1score])

model.fit(x={'transformer' : trainX}, y=trainY, batch_size=batch_size, epochs=10,
         validation_data=({'transformer' : testX}, testY))

score, acc, prec, recall, f1 = model.evaluate(x={'transformer' : testX}, y=testY, batch_size=batch_size)

print('Test score: {:.4f}'.format(score))
print('Test accuracy: {:.4f}'.format(acc))
print('Test p/r (f1): {:.2f}/{:.2f} ({:.2f})'.format(prec, recall, f1))

model.save(os.path.join(args.model_path,'transformer.h5'))
