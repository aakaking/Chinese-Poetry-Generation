# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:28:29 2018

@author: Anan
"""
import sys
import os
import json
from segment import *

def read(path):
    file = open(path, "rb")
    fileJson = json.load(file)
    return fileJson

def save(data,path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def get_stopwords(filename = "D:/Code/Jupyter/ChinesePoem/stopwords.txt"):
    stopwords_dic = open(filename, encoding= 'utf-8')
    stopwords = stopwords_dic.readlines()
    stopwords = [w.strip() for w in stopwords]
    stopwords_dic.close()
    return stopwords

def text_rank(adjlist):
    damp = 0.85
    scores = dict((word,1.0) for word in adjlist)
    try:
        for i in range(10000):
            print("[TextRank] Start iteration %d ..." %i)
            sys.stdout.flush()
            cnt = 0
            new_scores = dict()
            for word in adjlist:
                new_scores[word] = (1-damp)+damp*sum(adjlist[other][word]*scores[other] \
                        for other in adjlist[word])
                if scores[word] != new_scores[word]:
                    cnt += 1
            print("Done (%d/%d)" %(cnt, len(scores)))
            if 0 == cnt:
                break
            else:
                scores = new_scores
        print("TextRank is done.")
    except KeyboardInterrupt:
        print("\nTextRank is interrupted.")
    words = sorted([(word,score) for word,score in scores.items()],key=lambda x:x[1],reverse = True)
    return words

def co_occurrence():
    segmenter = Segmenter()
    stopwords = get_stopwords()
    print("Start TextRank over the selected quatrains ...")
    quatrains = read('mini_quatrains.json')
    adjlist = dict()
    for idx, poem in enumerate(quatrains):
        if 0 == (idx+1)%10000:
            print("[TextRank] Scanning %d/%d poems ..." %(idx+1, len(quatrains)))
        for sentence in poem['paragraphs']:
            segs = list(filter(lambda word: word not in stopwords,
                    segmenter.cut(sentence)))
            for seg in segs:
                if seg not in adjlist:
                    adjlist[seg] = dict()
            for i, seg in enumerate(segs):
                for _, other in enumerate(segs[i+1:]):
                    if seg != other:
                        adjlist[seg][other] = adjlist[seg][other]+1 \
                                if other in adjlist[seg] else 1.0
                        adjlist[other][seg] = adjlist[other][seg]+1 \
                                if seg in adjlist[other] else 1.0
    for word in adjlist:
        w_sum = sum(weight for other, weight in adjlist[word].items())
        for other in adjlist[word]:
            adjlist[word][other] /= w_sum  
    return adjlist

def get_word_ranks():
    ranks = text_rank(co_occurrence())
    rank_words = dict((pair[0], idx) for idx, pair in enumerate(ranks))
    save(rank_words,'rank_words.json')
    return rank_words

if __name__ == '__main__':
    ranks = get_word_ranks()