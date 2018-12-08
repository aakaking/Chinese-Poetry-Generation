# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:56:52 2018

@author: Anan
"""
import random
import json
from segment import *

def read(path):
    file = open(path, "rb")
    fileJson = json.load(file)
    return fileJson

def save(data,path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
 
       
start = '^'
end = '$'
        
def _gen_train_data():
    segmenter = Segmenter()
    poems = read('mini_quatrains.json')
    random.shuffle(poems)
    ranks = read('rank_words.json')
    print("Generating training data ...")
    plan_data = []
    gen_data = []
    for idx, poem in enumerate(poems):
        sentences = poem['paragraphs']
        if len(sentences) != 4:
            continue # Only consider quatrains.
        flag = True
        context = start
        gen_lines = []
        keywords = []
        for sentence in sentences:
            if len(sentence) != 7:
                flag = False
                break
            words = list(filter(lambda seg: seg in ranks, segmenter.cut(sentence)))
            if len(words) == 0:
                flag = False
                break
            keyword = reduce(lambda x,y: x if ranks[x] < ranks[y] else y, words)
            gen_line = sentence + end + '\t' + keyword + '\t' + context + '\n'
            gen_lines.append(gen_line)
            keywords.append(keyword)
            context += sentence + end
        if flag:
            plan_data.append('\t'.join(keywords) + '\n')
            gen_data.extend(gen_lines)
    save(gen_data,"train_data")
    save(plan_data,"train_keyword")
    
    return gen_data,plan_data

def row_to_id(word_to_id,row):
    return [word_to_id[word] for word in row]


gen_data, plan_data = _gen_train_data()
data_to_id = []
for line in gen_data:
    sample = []
    toks = line.strip().split('\t')    
    sentence = row_to_id(word_to_id,toks[0])
    keyword = row_to_id(word_to_id,toks[1])
    context = row_to_id(word_to_id,toks[2])
    #sample.append(keyword)
    sample.append(keyword+context)
    sample.append(sentence)
    data_to_id.append(sample)

    
'''
if __name__ == '__main__':
    row, kw_row = _gen_train_data()
    '''