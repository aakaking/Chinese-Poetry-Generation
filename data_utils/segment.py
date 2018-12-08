# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 19:53:23 2018

@author: Anan
"""

import jieba
from quatrains import *

def gen_sxhy_corrpus():
    sxhy_corpus = []
    with open("D:/Code/Jupyter/ChinesePoem/shixuehanying.txt", 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        while line:
            if line.startswith('<begin>'):
                tag = line.split('\t')[2]
            elif not line.startswith('<end>'):
                toks = line.split('\t')
                if len(toks) == 3:
                    toks = toks[2].split(' ')
                    tok_list = []
                    for tok in toks:
                        if len(tok) < 4:
                            tok_list.append(tok)
                        else:
                            tok_list.extend(jieba.lcut(tok, HMM=True))
                        sxhy_corpus += tok_list
            line = f.readline().strip()
    with open("D:/Code/Jupyter/ChinesePoem/sxhy_corpus.txt", 'w', encoding='utf-8') as f:
        for word in set(sxhy_corpus):
            f.write(word+'\n')
            
            
def get_sxhy_corpus():
    sxhy_corpus = []
    with open("D:/Code/Jupyter/ChinesePoem/sxhy_corpus.txt", 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            if is_CN_char(line.strip()):
                sxhy_corpus.append(line.strip())
            line = f.readline()
    return sxhy_corpus

#BM
class Segmenter:
    def __init__(self):
        self.dictionary = get_sxhy_corpus()
        self.maximum = 5
        
    def cut(self,text):
        
        def IMM_cut(text):
            result = []
            index = len(text)
            while index > 0:
                word = None
                for size in range(self.maximum, 0, -1):
                    if index - size < 0:
                        continue
                    piece = text[(index - size):index]
                    if piece in self.dictionary:
                        word = piece
                        result.append(word)
                        index -= size
                        break
                if word is None:
                    index -= 1
            return result[::-1]

        def FMM_cut(text):
            result = []
            index = 0
            while index < len(text):
                word = None
                for size in range(0,self.maximum,1):
                    if index+size > len(text):
                        continue
                    piece = text[index:index+size]
                    if piece in self.dictionary:
                        word = piece
                        result.append(word)
                        index += size
                        break
                if word is None:
                    index += 1
            return result
        
        IMM = IMM_cut(text)
        FMM = FMM_cut(text)
        if len(IMM) < len(FMM):
            return IMM
        if len(IMM) == len(FMM):
            return IMM
        if len(IMM) > len(FMM):
            return FMM