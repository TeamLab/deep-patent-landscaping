import os
import argparse
import pandas as pd
import numpy as np
import pickle
from dataset import PatentLandscaping
from utils import get_embedding_layer
from models import trf_diff_model
from keras.models import load_model
from keras import optimizers
from sklearn.utils import class_weight
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, roc_auc_score)
from callbacks import Metrics, Checkpoint


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='data name')
    parser.add_argument('--data_path', default='./data', help='data name')
    parser.add_argument('--word_embedding', default='w2v', help='word embedding')
    parser.add_argument('--code_embedding', default='graph', help='code embeddings')
    parser.add_argument('--embedding_path', default='embeddings', help='embeddings path')
    parser.add_argument('--model_path', default='./model', help='data name')
    parser.add_argument('--max_length', type=int, default=128, help='max sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--num_head', type=int, default=8, help='number of epochs')
    parser.add_argument('--num_layer', type=int, default=6, help='number of epochs')
    parser.add_argument('--hidden_size', type=int, default=512, help='number of epochs')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parameter_parser()
    
    if not os.path.exists(os.path.join(args.model_path, args.data)):
        os.makedirs(os.path.join(args.model_path, args.data))
    if not os.path.exists(os.path.join(args.embedding_path, args.word_embedding)):
        os.makedirs(os.path.join(args.embedding_path, args.word_embedding))
    if not os.path.exists(os.path.join(args.embedding_path, args.code_embedding)):
        os.makedirs(os.path.join(args.embedding_path, args.code_embedding))
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
    exit()
    pl = PatentLandscaping(args)
    pl.load_data()
    
    cpc_embedding_layer = get_embedding_layer(pl.cpc2idx, pl.cpc_vectors)
    ipc_embedding_layer = get_embedding_layer(pl.ipc2idx, pl.ipc_vectors)
    uspc_embedding_layer = get_embedding_layer(pl.uspc2idx, pl.uspc_vectors)
    
    model = trf_diff_model(args, pl.word_vectors, cpc_embedding_layer, ipc_embedding_layer, uspc_embedding_layer,
                              pl.max_cpc_len, pl.max_ipc_len, pl.max_uspc_len)
    print(model.summary())
    class_weights = class_weight.compute_class_weight('balanced', np.unique(pl.train_y),
                                                         np.argmax(pl.train_y, axis=1))
    metrics = Metrics('all')
    checkpoint = Checkpoint(metrics, args, 'all')
    adam = optimizers.Adam(lr=0.0001, epsilon=1e-8)
    
    model.compile(loss='categorical_crossentropy',
                     optimizer=adam,
                     metrics=['accuracy'])
    
    model.fit(x=pl.train, y=pl.train_y, batch_size=args.batch_size, epochs=args.epoch,
                 validation_data=(pl.val, pl.val_y), callbacks=[metrics, checkpoint],
                 class_weight=class_weights)
	
    print('Load best model')
    model = load_model(os.path.join(args.model_path, args.data, args.data+'_model.h5'),
                          custom_objects=get_custom_objects())
    
    score, acc = model.evaluate(x=pl.test, y=pl.test_y, batch_size=args.batch_size)
    y_pred = np.argmax(model.predict(pl.test, batch_size=args.batch_size), axis=1)
    y_true = np.argmax(pl.test_y, axis=1)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    
    print('Test score: {:.4f}'.format(score))
    print('Test accuracy: {:.4f}'.format(acc))
    print('Test average precision: {:.4f}'.format(average_precision))
    print('Test p/r (f1): {:.4f}/{:.4f} ({:.4f})'.format(precision, recall, f1))
    print('Test auc score : {:.4f}'.format(auc_score))

    
