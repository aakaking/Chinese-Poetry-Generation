import math
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pylab
from collections import Counter

# Seq2Seq Items
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.layers.core import Dense

vocab_size= len(word_to_id)
num_units = 128
input_size = 128
batch_size = 16
source_sequence_length=28
target_sequence_length=8
decoder_type = 'attention' # could be basic or attention
sentences_to_read = 50000



#Loading vocabularies -- > word_to_id

#Loading Sentences -- > row

#Add the special tokens and make all sentences same length (for batch-processing)

def row_to_id(word_to_id,row):
    row_to_id = []
    for r in row:
        sentence_to_id = []
        for s in r:
            sentence_to_id.append([word_to_id[word] for word in s])
        row_to_id.append(sentence_to_id)
    return row_to_id

def alignment(row_to_id,max_kw):
    aligned_row = []
    for row in row_to_id:
        alignment = []
        sentence = [0]*7
        sentence[:len(row[0])] = row[0]
        kw = [0]*max_kw
        kw[:len(row[1])] = row[1]
        alignment.append(sentence)
        alignment.append(kw)
        aligned_row.append(alignment)
    return aligned_row

def get_in_out(row):
    X = []
    Y = []
    for i in range(0,len(row),4):
        X.append(row[i][1]+[0])
        X.append(row[i+1][1]+[0]+row[i][0]+[0])
        X.append(row[i+2][1]+[0]+row[i][0]+[0]+row[i+1][0]+[0])
        X.append(row[i+3][1]+[0]+row[i][0]+[0]+row[i+1][0]+[0]+ row[i+2][0]+[0])
        Y.append(row[i][0]+[0])
        Y.append(row[i+1][0]+[0])
        Y.append(row[i+2][0]+[0])
        Y.append(row[i+3][0]+[0])
    return X,Y

def pad_sequences(sentence,maxlen):
    features = np.zeros((len(sentence), maxlen), dtype=int)
    for i,s in enumerate(sentence):
        features[i, :len(s)] = np.array(s)        
    return features

  
row2id = row_to_id(word_to_id,row)
aligned_row = alignment(row2id,3)
X,Y = get_in_out(aligned_row)
X_train = pad_sequences(X,source_sequence_length)
Y_train = pad_sequences(Y,target_sequence_length)

train_inp_lengths = np.array([len(x) for x in X], dtype=np.int32)
train_out_lengths = np.array([len(y) for y in Y], dtype=np.int32)



#Batch Data Generator

input_size = 128

class DataGeneratorMT(object):
    
    def __init__(self,batch_size,num_unroll,is_source):
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._cursor = [0 for offset in range(self._batch_size)]
        
        
        self._src_word_embeddings = np.load('embedding_matrix.npy')
        
        self._tgt_word_embeddings = np.load('embedding_matrix.npy')
        
        self._sent_ids = None
        
        self._is_source = is_source
        
                
    def next_batch(self, sent_ids, first_set):
        
        if self._is_source:
            max_sent_length = source_sequence_length
        else:
            max_sent_length = target_sequence_length
        batch_labels_ind = []
        batch_data = np.zeros((self._batch_size),dtype=np.float32)
        batch_labels = np.zeros((self._batch_size),dtype=np.float32)
        
        for b in range(self._batch_size):
            
            sent_id = sent_ids[b]
            
            if self._is_source:
                sent_text = train_inputs[sent_id]
                             
                batch_data[b] = sent_text[self._cursor[b]]
                batch_labels[b]=sent_text[self._cursor[b]+1]

            else:
                sent_text = train_outputs[sent_id]
                
                # We cannot avoid having two different embedding vectors for <s> token
                # in soruce and target languages
                # Therefore, if the symbol appears, we always take the source embedding vector
                if sent_text[self._cursor[b]]!=src_dictionary['<s>']:
                    batch_data[b] = sent_text[self._cursor[b]]
                else:
                    batch_data[b] = sent_text[self._cursor[b]]
                batch_labels[b] = sent_text[self._cursor[b]+1]

            self._cursor[b] = (self._cursor[b]+1)%(max_sent_length-1)
                                    
        return batch_data,batch_labels
        
    def unroll_batches(self,sent_ids):
        
        if sent_ids is not None:
            
            self._sent_ids = sent_ids
            
            #if self._is_source:
                # we dont star at the very beginning, becaues the very beginning is a bunch of </s> symbols.
                # so we start from the middel s.t we get a minimum number of </s> symbols in our training data
                # this is only needed for source language
                #self._cursor = ((start_indices_for_bins[bin_id][self._sent_ids]//self._num_unroll)*self._num_unroll).tolist()
            #else:
            self._cursor = [0 for _ in range(self._batch_size)]
                
        unroll_data,unroll_labels = [],[]
        inp_lengths = None
        for ui in range(self._num_unroll):
            # The first batch in any batch of captions is different
            if self._is_source:
                data, labels = self.next_batch(self._sent_ids, False)
            else:
                data, labels = self.next_batch(self._sent_ids, False)
                    
            unroll_data.append(data)
            unroll_labels.append(labels)
            inp_lengths = train_inp_lengths[sent_ids]
        return unroll_data, unroll_labels, self._sent_ids, inp_lengths
    
    def reset_indices(self):
        self._cursor = [0 for offset in range(self._batch_size)]
        
    
    
# Inputs Outputs Masks

tf.reset_default_graph()

enc_train_inputs = []
dec_train_inputs = []

# Need to use pre-trained word embeddings
encoder_emb_layer = tf.convert_to_tensor(np.load('embedding_matrix.npy'),tf.float32)
decoder_emb_layer = tf.convert_to_tensor(np.load('embedding_matrix.npy'),tf.float32)

# Defining unrolled training inputs
for ui in range(source_sequence_length):
    enc_train_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],name='enc_train_inputs_%d'%ui))

dec_train_labels=[]
dec_label_masks = []
for ui in range(target_sequence_length):
    dec_train_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],name='dec_train_inputs_%d'%ui))
    dec_train_labels.append(tf.placeholder(tf.int32, shape=[batch_size],name='dec-train_outputs_%d'%ui))
    dec_label_masks.append(tf.placeholder(tf.float32, shape=[batch_size],name='dec-label_masks_%d'%ui))
    
encoder_emb_inp = [tf.nn.embedding_lookup(encoder_emb_layer, src) for src in enc_train_inputs]
encoder_emb_inp = tf.stack(encoder_emb_inp)

decoder_emb_inp = [tf.nn.embedding_lookup(decoder_emb_layer, src) for src in dec_train_inputs]
decoder_emb_inp = tf.stack(decoder_emb_inp)

enc_train_inp_lengths = tf.placeholder(tf.int32, shape=[batch_size],name='train_input_lengths')
dec_train_inp_lengths = tf.placeholder(tf.int32, shape=[batch_size],name='train_output_lengths')

#Encoder
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

initial_state = encoder_cell.zero_state(batch_size, dtype=tf.float32)

encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_emb_inp, initial_state=initial_state,
    sequence_length=enc_train_inp_lengths, 
    time_major=True, swap_memory=True)

#Decoder
# Build RNN cell
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

projection_layer = Dense(units=vocab_size, use_bias=True)

# Helper
helper = tf.contrib.seq2seq.TrainingHelper(
    decoder_emb_inp, [target_sequence_length-1 for _ in range(batch_size)], time_major=True)


# Decoder
if decoder_type == 'basic':
    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, encoder_state,
        output_layer=projection_layer)
    
elif decoder_type == 'attention':
    decoder = tf.contrib.seq2seq.BahdanauAttention(
        decoder_cell, helper, encoder_state,
        )
# Dynamic decoding
outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
    decoder, output_time_major=True,
    swap_memory=True
)


attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size, memory=encoder_outputs,
                                                         memory_sequence_length=encoder_inputs_length)
#attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.rnn_size, memory=encoder_outputs, memory_sequence_length=encoder_inputs_length)
# 定义decoder阶段要是用的LSTMCell，然后为其封装attention wrapper
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                                   attention_layer_size=self.rnn_size, name='Attention_Wrapper')

#定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
output_layer = tf.layers.Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))






