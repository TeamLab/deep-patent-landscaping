import os
import random
import argparse
import pickle
import ast
import json
from keras.utils import to_categorical
from keras.preprocessing import sequence
from gensim.models.word2vec import Word2Vec
from keras_multi_head import MultiHeadAttention
from keras_layer_normalization import LayerNormalization
from keras_position_wise_feed_forward import FeedForward
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, ReLU, ELU, Concatenate
from keras.layers.core import Dropout
from transformer.keras_metrics import precision, recall, f1score
from keras import optimizers
from transformer.tokenizer import *
from utils import *
from transformer.transformer import PooledOutput
from keras.models import load_model
from gcn.graph import GraphConvolution
from gcn.utils import *
import pdb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sequence_len = 128
batch_size = 64
dropout_pct = 0.1
num_epochs = 20
random_seed = 1313


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='data folder path')
    parser.add_argument('--model_path', help='model folder path')
    parser.add_argument('--data', help='data name')
    parser.add_argument('--graph_model', help='refs cpc ipc uspc', default='refs cpc ipc uspc')
    args = parser.parse_args()

    return args


def load_graph_embedding(model_path, data_path):
    graph_model = load_model(model_path, custom_objects={'GraphConvolution': GraphConvolution})
    X, A, _, code2idx = pickle.load(open(data_path, 'rb'))
    adjacency_matrix = sp.csr_matrix(A)
    adjacency_matrix = preprocess_adj(adjacency_matrix, True)
    X /= X.sum(1).reshape(-1, 1)
    graph_data = [X, adjacency_matrix]
    graph_layer = Model(inputs=graph_model.input, outputs=graph_model.get_layer('embedding').output)
    embed = graph_layer.predict(graph_data, batch_size=X.shape[0])

    return embed, code2idx


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

    return word2idx, idx2word


def load_transformer_layer(filepath):
    model = load_model(filepath=filepath,
                       custom_objects={'MultiHeadAttention': MultiHeadAttention,
                                       'LayerNormalization': LayerNormalization,
                                       'FeedForward': FeedForward,
                                       'PooledOutput': PooledOutput,
                                       'precision': precision,
                                       'recall': recall,
                                       'fmeasure': f1score})
    model_layer = Model(inputs=model.input, outputs=model.get_layer('pooled_output_1').output)
    return model_layer


def define_model(trainGraph, trainX):
    graph_inputs = []
    graph_layer = []

    for idx in range(len(trainGraph)):
        graph_input = Input(shape=(trainGraph[0].shape[-1],), name='graph_{}'.format(idx))
        graph_inputs.append(graph_input)
        graph = BatchNormalization()(graph_input)
        graph_layer.append(graph)

    transformer_input = Input(shape=(trainX.shape[1],), name='transformer')

    model_inputs = [*graph_layer, transformer_input]

    final_layer = Concatenate(name='concatenated')(model_inputs)
    output = Dense(256)(final_layer)
    output = Dropout(dropout_pct)(output)
    output = BatchNormalization()(output)
    output = ReLU()(output)
    output = Dense(2, activation='sigmoid')(output)

    model = Model(inputs=[*graph_inputs, transformer_input],
                  outputs=output, name='model')
    return model


if __name__ == "__main__":
    args = parse_args()

    word_vector = Word2Vec.load(os.path.join('models', 'w2v_512.model')).wv
    word2idx, idx2word = create_word_vocab(word_vector)

    transformer_path = os.path.join(args.model_path, 'transformer.h5')
    transformer_layer = load_transformer_layer(transformer_path)

    data = pd.read_csv(args.data, index_col='Unnamed: 0').drop_duplicates()
    data = data.sample(frac=1).reset_index(drop=True)
    origin_data = data.copy()

    target_graph = []
    if 'cpc' in args.graph_model.split():
        print("cpc")
        cpc_embed, cpc2idx = load_graph_embedding(os.path.join(args.model_path, 'cpc_model.h5'),
                                                  os.path.join(args.data_path, 'cpc.pkl'))
        data.cpc_code = data.cpc_code.apply(lambda x: embed_mean_vector(ast.literal_eval(x), cpc2idx, cpc_embed))
        target_graph.append(np.vstack(data.cpc_code.values))
    if 'uspc' in args.graph_model:
        print("uspc")
        uspc_embed, uspc2idx = load_graph_embedding(os.path.join(args.model_path, 'uspc_model.h5'),
                                                    os.path.join(args.data_path, 'uspc.pkl'))
        data.uspc_code = data.uspc_code.apply(lambda x: embed_mean_vector(ast.literal_eval(x), uspc2idx, uspc_embed))
        target_graph.append(np.vstack(data.uspc_code.values))
    if 'ipc' in args.graph_model:
        print("ipc")
        ipc_embed, ipc2idx = load_graph_embedding(os.path.join(args.model_path, 'ipc_model.h5'),
                                                  os.path.join(args.data_path, 'ipc.pkl'))
        data.ipc_code = data.ipc_code.apply(lambda x: embed_mean_vector(ast.literal_eval(x), ipc2idx, ipc_embed))
        target_graph.append(np.vstack(data.ipc_code.values))
    if 'refs' in args.graph_model:
        print("refs")
        refs_embed, refs2idx = load_graph_embedding(os.path.join(args.model_path, 'refs_model.h5'),
                                                    os.path.join(args.data_path, 'refs.pkl'))
        data.refs = data.refs.apply(lambda x: embed_mean_vector(ast.literal_eval(x), refs2idx, refs_embed))
        target_graph.append(np.vstack(data.refs.values))

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    abstract_text = data.abstract_text
    tokenized_text = cleaning_sentences(abstract_text, string.punctuation)
    tokenized_indexed_text = get_indexed_token(tokenized_text, word2idx)
    padded_x = sequence.pad_sequences(sequences=tokenized_indexed_text,
                                      maxlen=sequence_len,
                                      padding='post', truncating='post')
    labels_indexed = to_categorical(data.valid.astype(int).values).astype(int)

    text_embed = transformer_layer.predict(padded_x, batch_size=64)
    random.seed(random_seed)

    indexes = list(range(len(labels_indexed)))
    #random.shuffle(indexes)
    split_val = int(len(data) * 0.8)

    train_index, test_index = indexes[:split_val], indexes[split_val:]
    trainX, testX = text_embed[train_index], text_embed[test_index]
    trainY, testY = labels_indexed[train_index], labels_indexed[test_index]
    trainGraph, testGraph = \
        [graph[train_index] for graph in target_graph], [graph[test_index] for graph in target_graph]

    model = define_model(trainGraph, trainX)

    adam = optimizers.Adam(lr=0.0001, epsilon=1e-8)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy', precision, recall, f1score])
    model.summary()
    model.fit(x=[*trainGraph, trainX],
              y=trainY, batch_size=batch_size, epochs=num_epochs,
              validation_data=([*testGraph, testX], testY))

    origin_data.to_csv('trained/origin_'+args.data.split('/')[-1])
    # store incorrect data
    pred = model.predict(x=[*testGraph, testX], batch_size=batch_size).argmax(axis=1)
    false_index = np.where((pred == np.argmax(testY, axis=1))==False)[0]# + split_val
    if len(test_index) > 0:
        false_index = false_index + split_val
        false_value = origin_data.iloc[false_index]
        false_value['target'] = args.graph_model
        with open("trained/" + args.data.split('/')[-1][:-4] + ".json", 'a') as f:
            for x in false_value.to_dict('records'):
                json.dump(x, f)
                f.write("\n")
    
    pred = model.predict(x=[*trainGraph, trainX], batch_size=batch_size).argmax(axis=1)
    false_index = np.where((pred == np.argmax(trainY, axis=1))==False)[0]# + split_val
    if len(train_index) > 0:
        false_index = false_index
        false_value = origin_data.iloc[false_index]
        false_value['target'] = args.graph_model
        with open("trained/" + args.data.split('/')[-1][:-4] + ".json", 'a') as f:
            for x in false_value.to_dict('records'):
                json.dump(x, f)
                f.write("\n")
    
    # save model 
    model_json = model.to_json()
    with open("trained/{}_{}.json".format(args.data.split('/')[-1][:-4], args.graph_model), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("trained/{}_{}.h5".format(args.data.split('/')[-1][:-4], args.graph_model))

    # save score
    results = dict()
    results.update({"data":args.data,
                    "targets": args.graph_model})

    train_score, train_acc, train_prec, train_recall, train_f1 = \
        model.evaluate(x=[*trainGraph, trainX], y=trainY, batch_size=batch_size)
    results.update({"train_loss": train_score,
                    "train_acc": train_acc,
                    "train_prec": train_prec,
                    "train_recall": train_recall,
                    "train_f1": train_f1})
    
    score, acc, prec, recall, f1 = \
        model.evaluate(x=[*testGraph, testX], y=testY, batch_size=batch_size)
    results.update({"test_loss": score,
                    "test_acc": acc,
                    "test_prec": prec,
                    "test_recall": recall,
                    "test_f1": f1})
    with open("trained/results.json", 'a') as f:
        json.dump(results, f)
        f.write("\n")
    
    print('Test score: {:.4f}'.format(score))
    print('Test accuracy: {:.4f}'.format(acc))
    print('Test p/r (f1): {:.2f}/{:.2f} ({:.2f})'.format(prec, recall, f1))
