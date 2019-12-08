from keras.layers import (Input, Lambda, Dense, Concatenate,
                          Dropout, Embedding, Softmax)
from layers import get_encoders
from keras.models import Model
import keras.backend as K


def trf_diff_model(args, word_vectors, cpc_layer, ipc_layer,
                    uspc_layer, cpc_len, ipc_len, uspc_len):
    cpc_input = Input(shape=(cpc_len,), name='cpc')
    cpc = cpc_layer(cpc_input)
    cpc = Lambda(lambda x: K.mean(x, axis=1))(cpc)
    cpc = Dense(256)(cpc)
    
    ipc_input = Input(shape=(ipc_len,), name='ipc')
    ipc = ipc_layer(ipc_input)
    ipc = Lambda(lambda x : K.mean(x, axis=1))(ipc)
    ipc = Dense(128)(ipc)
    
    uspc_input = Input(shape=(uspc_len,), name='uspc')
    uspc = uspc_layer(uspc_input)
    uspc = Lambda(lambda x : K.mean(x, axis=1))(uspc)
    uspc = Dense(128)(uspc)
    
    codes = Concatenate(name='codes')([cpc, ipc, uspc])
    codes = Dense(512, activation='tanh')(codes)
    codes = Dropout(args.dropout)(codes)
    
    abstract_input = Input(shape=(args.max_length,), name='abstract')
    abstract = Embedding(input_dim=word_vectors.vectors.shape[0],
                            output_dim=word_vectors.vector_size,
                            weights=[word_vectors.vectors],
                            input_length=args.max_length,
                            trainable=True)(abstract_input)
    abstract = get_encoders(args.num_layer, abstract, args.num_head, args.hidden_size, dropout_rate=args.dropout)
    abstract = Dropout(args.dropout)(abstract)
    
    model_inputs = [codes, abstract]
    
    output = Concatenate(name='concat')(model_inputs)
    output = Dense(2)(output)
    output = Softmax()(output)
    
    model = Model(inputs=[cpc_input, ipc_input, uspc_input, abstract_input], outputs=output, name='model')
    return model
