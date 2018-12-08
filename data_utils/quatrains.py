# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 17:26:28 2018

@author: Anan
"""

import json
from opencc import OpenCC
from functools import reduce

def is_CN_char(ch):
    return ch >= u'\u4e00' and ch <= u'\u9fa5'

def merge_sentences(line):
    sentences = []
    i = 0
    for j in range(len(line)+1):
        if j == len(line) or line[j] in [u'，', u'。', u'！', u'？', u'、']:
            if i < j:
                sentence = u''.join(filter(is_CN_char, line[i:j]))
                sentences.append(sentence)
            i = j+1
    return sentences

def convert2simple(word):
    openCC = OpenCC('tw2sp')
    return openCC.convert(word)

def split_sentence(line):
    sentences = []
    i = 0
    for j in range(len(line)):
        if line[j] in [u'，', u'。', u'！', u'？', u'、']:
            if i < j:
                sentence = u''.join(filter(is_CN_char, line[i:j]))
                sentences.append(sentence)
            i = j+1
    return sentences

def get_quatrains(content):    
    quatrain = []
    for poem in content:
        sentences = poem['paragraphs']
        if len(sentences) == 4 and \
            (len(sentences[0]) == 5 or len(sentences[0]) == 7) and \
            reduce(lambda x, sentence: x and len(sentence) == len(sentences[0]),
                    sentences[1:], True):
            quatrain.append(poem)
    return quatrain
    
def save(data,path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def read(path):
    file = open(path, "rb")
    fileJson = json.load(file)
    return fileJson

def process_peom(content):
    for poem in content:
        if len(poem['paragraphs']) > 0:
            s = merge_sentences(poem['paragraphs'])
            sentence = convert2simple(s[0])
            poem['paragraphs'] = split_sentence(sentence) 
    return content


if __name__ == '__main__':
    path = "D:/Code/Jupyter/ChinesePoem/chinese-poetry-master/json/poet.tang.1000.json"
    content = read(path)
    content = process_peom(content)
    content = get_quatrains(content)
    save(content,'mini_quatrains.json')


    