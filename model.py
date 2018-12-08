from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM,Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from attention_decoder import AttentionDecoder

kw = []
for row in kw_row:
    for w in row:
        kw.append(w)
kw_len = [len(w) for w in kw]
max_kw = max(kw_len)

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

maxlen = 25 + max_kw    
row2id = row_to_id(word_to_id,row)
aligned_row = alignment(row2id,3)
X,Y = get_in_out(aligned_row)
X_train = pad_sequences(X,maxlen)
Y_train = pad_sequences(Y,maxlen)

def embedding_encode(sequence, embedding):
	encoding = list()
	for value in sequence:
		encoding.append(embedding[value])
	return np.array(encoding)
# prepare data for the LSTM
def get_pair(X_train, Y_train):
	X = embedding_encode(X_train, embedding)
	y = embedding_encode(Y_train, embedding)
	X = X.reshape((1, X.shape[0], X.shape[1]))
	y = y.reshape((1, y.shape[0], y.shape[1]))
	return X,y

vocab_size = len(word_to_id)
embed_size = 128

# define model
model = Sequential()
#model.add(Embedding(vocab_size, embed_size, weights=[embedding], input_length=maxlen, trainable=False))
model.add(Bidirectional(LSTM(150,return_sequences=True)))
model.add(AttentionDecoder(150,vocab_size))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

for epoch in range(5000):
    i = randint(0, len(row)-1)
	# generate new random sequence
    X,y = get_pair(X_train[i], Y_train[i])
	# fit model for one epoch on this sequence
    model.fit(X, Y_train[i], epochs=1, verbose=2)

def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

for _ in range(total):
	X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
	yhat = model.predict(X, verbose=0)
	if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
		correct += 1
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
# spot check some examples
for _ in range(10):
	X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
	yhat = model.predict(X, verbose=0)
	print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))
