import random, os, sys
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf
from keras import backend as K

try:
	from dataloader import TokenList, pad_to_longest
	# for transformer
except: pass

class LayerNormalization(Layer):

    def __init__(self, epsilon=1e-8, **kwargs):
        self.epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.mean(x, axis=-1, keepdims=True)

        return self.gamma * (x - mean) / ((std + self.epsilon)**(.5)) + self.beta

    def compute_output_shape(self, input_shape):

        return input_shape

class PooledOutput(Layer):

    def __init__(self, d_model=512, **kwargs):
        self.d_model = d_model
        super(PooledOutput, self).__init__(**kwargs)

    def build(self, input_shape):
        self.pool = self.add_weight(name='poolweight', shape=(self.d_model, self.d_model), initializer=RandomNormal(), trainable=True)
        super(PooledOutput, self).build(input_shape)

    def call(self, x):
        first_token = K.squeeze(x[:, 0:1, :], axis=1)

        return K.dot(first_token, self.pool)

    def compute_output_shape(self, input_shape):

        return (input_shape[0], input_shape[2])

class PositionalEncoding():

    def __init__(self, sequence_len=150, d_model=512, zero_pad=True, scale=True, **kwargs):
        self.d_model = d_model
        self.zero_pad = zero_pad
        self.scale = scale
        self.sequence_len = sequence_len

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def __call__(self, x):
        src_seq_input = Input(shape=(self.sequence_len,), dtype='int32')
        src_pos = Lambda(self.get_pos_seq)(src_seq_input)

        position_enc = np.array([[pos / np.power(10000, 2.*i/self.d_model) for i in range(self.d_model)]
                                if pos != 0 else np.zeros(self.d_model) for pos in range(self.sequence_len)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])

        position_embed = Embedding(self.sequence_len, self.d_model, trainable=False, weights=[position_enc])(src_pos)

        return Add()([x, position_embed])

class Mask():

    def __init__(self):
        pass

    def __call__(self, x):

        return x * K.expand_dims(K.sign(K.sum(K.abs(x), axis=-1)))

class MultiheadAttention():

    def __init__(self, d_model, num_heads, dropout):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_q = d_model // num_heads
        self.d_k = d_model // num_heads
        self.scale = np.sqrt(d_model // num_heads)
        self.dropout = Dropout(dropout)
        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []
        for _ in range(num_heads):
            self.qs_layers.append(TimeDistributed(Dense(self.d_q, use_bias=False)))
            self.ks_layers.append(TimeDistributed(Dense(self.d_k, use_bias=False)))
            self.vs_layers.append(TimeDistributed(Dense(self.d_k, use_bias=False)))
        self.layer_norm = LayerNormalization()
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        heads = []
        for i in range(self.num_heads):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](v)
            attn = Lambda(lambda x : K.batch_dot(x[0], x[1], axes=[2, 2]) / self.scale)([q, k])
            if mask is not None:
                masking = Lambda(lambda x : (-1e+10)*(1-x))(mask)
                attn = Add()([attn, mmask])
            attn = Activation('softmax')(attn)
            # attn = self.dropout(attn)
            head = Lambda(lambda x : K.batch_dot(x[0], x[1]))([attn, v])
            heads.append(head)
        head = Concatenate()(heads)

        outputs = self.w_o(head)
        outputs = self.dropout(outputs)
        outputs = Add()([outputs, q])

        return self.layer_norm(outputs)

class PositionwiseFeedForward():

    def __init__(self, num_units, dropout):
        self.w_1 = Conv1D(num_units[0], 1, activation='relu')
        self.w_2 = Conv1D(num_units[1], 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])

        return self.layer_norm(output)

class Transformer():

    def __init__(self, d_model=512, num_heads=8, num_blocks=6, num_units=[2048, 512], dropout=0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_units = num_units
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.self_attn_layer = MultiheadAttention(d_model, num_heads, dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(num_units, dropout)

    def __call__(self, x):
        enc_output = x
        for i in range(self.num_blocks):
            enc_output = self.self_attn_layer(enc_output, enc_output, enc_output)
            enc_output = self.pos_ffn_layer(enc_output)

        return enc_output
