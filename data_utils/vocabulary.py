# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:08:08 2018

@author: Anan
"""

import json
from collections import Counter

def read(path):
    file = open(path, "rb")
    fileJson = json.load(file)
    return fileJson

def get_all_char(poems):
    vocab = set()
    for idx, poem in enumerate(poems):
        for sentence in poem['paragraphs']:
            for char in sentence:
                if char not in [u'，', u'。', u'！', u'？', u'、']:
                    vocab.add(char)
    return vocab

def char_to_id(vocab):
    counts = Counter(vocab)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab = ['<pad>'] + [u'^'] + [u'$']+ vocab
    word_to_id = { word : i for i, word in enumerate(vocab)}    
    id_to_word = {i:word for i,word in enumerate(vocab)}
    return word_to_id, id_to_word

if __name__ == '__main__':
    poems = read('mini_quatrains.json')
    vocab = get_all_char(poems)
    word_to_id,id_to_word = char_to_id(vocab)