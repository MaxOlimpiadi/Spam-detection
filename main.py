# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:55:35 2024

@author: zidan
"""

import nltk
import math
import numpy as np
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import spacy
nlp = spacy.load("en_core_web_sm")


#check if the lemma is empty (completely empty or consisting only of spaces, etc.)
def is_empty(lemma):
    for i in range(0, len(lemma)):
        if lemma[i] != ' ' and lemma[i] != '\n' and lemma[i] != '\t':
            return 0
        else:
            #print(str(lemma) + ' is empty')
            return 1

def preprocessing(text):
    doc = nlp(text)
    lem_text = []
    #lem_text = [token.lemma_ for token in doc]
    for i in range(0, len(doc)):
        lemma = doc[i].lemma_
        if (not is_empty(lemma)):
            lem_text.append(lemma.lower())
    return lem_text

def get_vocabulary(lines): #каждая "линия" - список, в котором каждый элемент - слово
    vocab = set()
    for i in range(0, len(lines)):
        vocab = set.union(vocab, set(lines[i]))
    vocab = list(vocab)
    vocab.sort()
    vocab_dict = dict.fromkeys(vocab)
    i = 0
    for w in vocab_dict.keys():
        vocab_dict[w] = i
        i += 1
    return vocab_dict


def get_freqs(vocab_dict, lines):
    freqs = []
    len_vocab = len(vocab_dict) #number of words in dictionary
    for i in range(0, len(lines)):
        freqs.append([0] * len_vocab)
    
    for i in range(0, len(lines)):
        for j in range(0, len(lines[i])):
            tmp = vocab_dict[lines[i][j]] #got the number of the current word in the dictionary (i.e. the column number in our matrix)
            freqs[i][tmp] += 1
    return freqs    


def inicialize_params(N):
    ww = []
    for i in range(0, len(vocab_dict)):
        ww.append(np.random.uniform(0, 1)) 
    return ww          


def evaluate_probab(ww, feature_vector):
    z = np.dot(ww, feature_vector)
    p = 1 / (1 + np.exp(-z))
    if p == 0:
        print("Bad: " + str(z))
        print("Exp(z) = " + str(np.exp(-z)) + ' division: ' + str(1 / 1 + (np.exp(z))))
    return p


#calculate the cost function for all data (take the arithmetic mean)
def calc_cost_func(y, probs):
    sum = 0
    for i in range(0, len(y)):
        sum += y[i] * math.log(probs[i]) + (1 - y[i]) * math.log(1 - probs[i])
    J = (-1) * (sum / len(y))
    print(J)
    return J
 


def reestimate_params(ww, alpha, y, probs, freqs):
    for i in range(0, len(ww)):
        sum = 0
        for j in range(0, len(freqs)):
            sum += (probs[j] - y[j]) * freqs[j][i]
        ww[i] -= alpha * (1 / len(y)) * sum 
    return ww
    
    
       
lem_texts = []
y = []
f = open("SMS_Spam_Corpus_big.txt", 'r')
lines = f.readlines()
for i in range(0, len(lines)):
    cur_line = lines[i]    #current message (line)
    if cur_line[-5:] == 'spam\n': #if there is a "spam" tag at the end..
        y.append(1)
        lines[i] = lines[i][:-6] #..then remove the "spam" tag and the comma from the end
    else: 
        y.append(0)
        lines[i] = lines[i][:-5] #remove the "ham" tag and the comma from the end
for i in range(0, len(lines)):
    lines[i] = preprocessing(lines[i])  #preprocessing every line (message)

vocab_dict = get_vocabulary(lines) #getting vocab
freqs = get_freqs(vocab_dict, lines) #vectorization

ww = inicialize_params(len(vocab_dict)) 


#TODO: move it into a separate function
#================ ITERATIVE PROCESS =====================#
J_OLD = 1000    #the initial value of the cost function
old_y_pred = [0] * len(y)   #initial values of predictions
alpha = 0.1     #step
N = 1000    #number of iterations
#y_pred = []
for i in range(0, N):
    y_pred = []
    probs = []
    for j in range(0, len(lines)):
        p = evaluate_probab(ww, freqs[j])
        probs.append(p)
        if  p >= 0.5: #if the estimated probability is greater than or equal to 0.5, .. 
            y_pred.append(1)                #..then consider the current message as spam
            #print(p)
        else:
            y_pred.append(0)
            #print(p)
    J = calc_cost_func(y, probs)
    if J > J_OLD:
        print("After iteration #" + str(i) + " the following predictions have been obtained:")
        print("New J: " + str(J) + " old j: " + str(J_OLD))
        #print(y_pred)
        break
    else:
        J_OLD = J
        ww = reestimate_params(ww, alpha, y, probs, freqs)
        for j in range(0, len(y)):
            old_y_pred[j] = y_pred[j]
print(y_pred)   #final predictions