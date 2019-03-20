from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from graph import *
from gcn_input_utils import *
import time
from utils import *
import pickle

import scipy.sparse as sp
import ast
import argparse

FILTER = 'localpool'
MAX_DEGREE = 2
SYM_NORM = True
NB_EPOCH = 200
PATIENCE = 10

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='data folder path')
parser.add_argument('--model_path', help='model folder path')
parser.add_argument('--data', help='data name')
args = parser.parse_args()

if not os.path.exists(args.model_path):
    os.mkdir(args.model_path)


file = open(os.path.join(args.data_path, args.data), 'rb')
data = pickle.load(file)

X = data[0]
A = data[1]
y = data[2]
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)
A = sp.csr_matrix(A)
A_ = preprocess_adj(A, True)
X /= X.sum(1).reshape(-1, 1)
support = 1
graph = [X, A_]

G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

X_in = Input(shape=(X.shape[1], ))

H = Dropout(0.5)(X_in)
H = GraphConvolution(128, support, activation='relu', kernel_regularizer=l2(5e-4), name='embedding')([H]+G)
H = Dropout(0.5)(H)
Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)

model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01))

wait = 0
preds = None
best_val_loss = 99999

for epoch in range(1, NB_EPOCH+1):
    t = time.time()

    model.fit(graph, y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

    preds = model.predict(graph, batch_size=A.shape[0])

    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val], [idx_train, idx_val])

    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.4f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
          "time= {:.4f}".format(time.time() - t))

    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

model.save('{}_model.h5'.format(os.path.join(args.model_path, args.data.split('.')[0])))

test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
