# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 21:08:38 2018

@author: Anan
"""
import os
import numpy as np
from numpy.random import uniform
from segment import Segmenter
from vocabulary import char_to_id
from gensim import models



def read(path):
    file = open(path, "rb")
    fileJson = json.load(file)
    return fileJson


def gen_word_embedding(ndim):
    vocab = read('vocab.json')
    word_to_id,id_to_word = char_to_id(vocab)
    vocab_size = len(word_to_id)
    ch_lists = []
    quatrains = read('mini_quatrains.json')
    for poem in quatrains:
        sentence = poem['paragraphs']
        for i in range(len(sentence[0])):
            ver_char = []
            for j in range(4):
                ver_char.append(sentence[j][i])
            ch_lists.append(ver_char)
        for s in sentence:
            ch_lists.append([char for char in s])
    model = models.Word2Vec(ch_lists, size = ndim, min_count = 5)
    model.save("word2vec.model")
    embedding_matrix = uniform(-1.0, 1.0, [vocab_size,ndim])
    for word, i in word_to_id.items():
        if word not in model.wv.vocab:
            continue
        else:
            embedding_matrix[i] = model[word]
    np.save('embedding_matrix.npy', embedding_matrix)
    return embedding_matrix
    

def get_word_embedding(ndim):
    if not os.path.exists('embedding_matrix'):
        gen_word_embedding(ndim)
    return np.load('embedding_matrix.npy')


if __name__ == '__main__':
    embedding = get_word_embedding(128)
    print("Size of embedding: (%d, %d)" %embedding.shape)