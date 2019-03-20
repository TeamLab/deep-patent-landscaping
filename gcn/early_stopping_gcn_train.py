import os
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from graph import *
from gcn_input_utils import *
import time
from utils import *
import pickle
import json
import scipy.sparse as sp
import argparse
from tensorflow.python.client import device_lib
import pdb
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

FILTER = 'localpool'
MAX_DEGREE = 2
SYM_NORM = True
NB_EPOCH = 200
PATIENCE = 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='data folder path')
    parser.add_argument('--model_path', help='model folder path')
    parser.add_argument('--data', help='data name')
    args = parser.parse_args()

    return args


def define_model(X, y, support=1):
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]
    X_in = Input(shape=(X.shape[1], ))
    H = Dropout(0.5)(X_in)
    H = GraphConvolution(128, support, activation='relu', kernel_regularizer=l2(5e-4), name='embedding')([H]+G)
    H = Dropout(0.5)(H)
    Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)

    model = Model(inputs=[X_in]+G, outputs=Y)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])

    return model


def train_model(model, graph, y, args):
    wait = 0
    preds = None
    best_val_loss = 99999
    y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

    for epoch in range(1, NB_EPOCH+1):
        t = time.time()
        model.fit(graph, y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)
        preds = model.predict(graph, batch_size=A.shape[0])
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val], [idx_train, idx_val])

        #print("Epoch: {:04d}".format(epoch),
        #      "train_loss= {:.4f}".format(train_val_loss[0]),
        #      "train_acc= {:.4f}".format(train_val_acc[0]),
        #      "val_loss= {:.4f}".format(train_val_loss[1]),
        #      "val_acc= {:.4f}".format(train_val_acc[1]),
        #      "time= {:.4f}".format(time.time() - t))

        if train_val_loss[1] < best_val_loss:
            best_val_loss = train_val_loss[1]
            wait = 0
        else:
            if wait >= PATIENCE:
                print('Epoch {}: early stopping'.format(epoch))
                break
            wait += 1

    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])

    print("train loss : {}, acc {}".format(train_val_loss[0], train_val_acc[0]))
    print("valid loss : {}, acc {}".format(train_val_loss[1], train_val_acc[1]))
    print("test  loss : {}, acc {}".format(test_loss[0], test_acc[0]))
    
    results = {"file": args.data_path,
                "target":args.data,
                "train_loss": "{}".format(train_val_loss[0]), 
                "test_loss": "{}".format(test_loss[0]), 
                "train_acc": "{}".format(train_val_acc[0]), 
                "test_acc": "{}".format(test_acc[0])
                }

    with open('results.json', 'a') as outfile:
        json.dump(results, outfile)
        outfile.write("\n")

    return model
    #print("Test set results:",
    #      "loss= {:.4f}".format(test_loss[0]),
    #      "accuracy= {:.4f}".format(test_acc[0]))


def create_input_graph(adjacency_matrix, feature_matrix):
    adjacency_matrix = sp.csr_matrix(adjacency_matrix)
    adjacency_matrix_ = preprocess_adj(adjacency_matrix, True)
    feature_matrix /= feature_matrix.sum(1).reshape(-1, 1)
    graph = [feature_matrix, adjacency_matrix_]
    return graph


if __name__ == "__main__":
    args = parse_args()
   
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    with open(os.path.join(args.data_path, args.data), 'rb') as f:
        X, A, y, _ = pickle.load(f)

    print(X.shape, y.shape)
    #if X.shape[0] > 30000:
    #    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    #    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    support = 1
    model = define_model(X, y, support=support)
    graph = create_input_graph(A, X)
    #model = train_model(model=model, graph=graph, y=y, args=args)
    y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)
    early_stopping = EarlyStopping(patience=50)
    model.fit(graph, y_train, sample_weight=train_mask, batch_size=A.shape[0], shuffle=False,
        epochs=NB_EPOCH, callbacks=[early_stopping], validation_data=(graph, y_val))
    preds = model.predict(graph, batch_size=A.shape[0])
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val], [idx_train, idx_val])
    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
    model.save('{}_model.h5'.format(os.path.join(args.model_path, args.data.split('.')[0])))

