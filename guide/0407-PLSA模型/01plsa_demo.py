#参考文献https://github.com/laserwave/plsa/blob/master/plsa.py

from numpy import zeros, int8, log
import numpy as np
from pylab import random
import jieba
import re
import codecs

# segmentation, stopwords filtering and document-word matrix generating
# [return]:
# N : number of documents
# M : length of dictionary
# word2id : a map mapping terms to their corresponding ids
# id2word : a map mapping ids to terms
# X : document-word matrix, N*M, each line is the number of terms that show up in the document
def preprocessing(datasetFilePath, stopwordsFilePath):
    
    # read the stopwords file
    file = codecs.open(stopwordsFilePath, 'r', 'utf-8')
    stopwords = [line.strip() for line in file] 
    file.close()
    
    # read the documents
    file = codecs.open(datasetFilePath, 'r', 'utf-8')
    documents = [document.strip() for document in file] 
    file.close()

    # number of documents
    N = len(documents)

    wordCounts = [];
    #apple apple apple apple apple apple apple apple apple banana  banana  grape
    #apple:9  banana:2  grape:1
    word2id = {}
    id2word = {}
    currentId = 0;
    # generate the word2id and id2word maps and count the number of times of words showing up in documents
    for document in documents:
        segList = jieba.cut(document)
        wordCount = {}
        for word in segList:
            word = word.lower().strip()
            if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords:               
                if word not in word2id.keys():
                    word2id[word] = currentId;
                    id2word[currentId] = word;
                    currentId += 1;
                if word in wordCount:
                    wordCount[word] += 1
                else:
                    wordCount[word] = 1
        wordCounts.append(wordCount);
    
    # length of dictionary
    M = len(word2id)  

    # generate the document-word matrix
    X = zeros([N, M], int8)
    for word in word2id.keys():
        j = word2id[word]
        for i in range(0, N):
            if word in wordCounts[i]:
                X[i, j] = wordCounts[i][word];    

    return N, M, word2id, id2word, X

N, M, word2id, id2word, X = preprocessing("my_corpus.txt","my_stop_words.txt")
print("N",N)
print("M",M)
print("word2id",word2id)
print("id2word",id2word)
print("X",X)
print("")

# --------------------------------------------------------------------------------
# N个文档
# K个话题
# M个单词

def initializeParameters():
    for i in range(0, N):
        normalization = sum(lamda[i, :])
        for j in range(0, K):
            lamda[i, j] /= normalization;

    for i in range(0, K):
        normalization = sum(theta[i, :])
        for j in range(0, M):
            theta[i, j] /= normalization;
N = N
K =2
M = M

# lamda[i, j] : p(zj|di)
lamda = random([N, K])

# theta[i, j] : p(wj|zi)
theta = random([K, M])

# p[i, j, k] : p(zk|di,wj)
p = zeros([N, M, K])

initializeParameters()

print("init lamda",lamda)
print("init theta",theta)
#print("p",p)
print("")

# --------------------------------------------------------------------------------
def EStep():
    for i in range(0, N):#修改p[i,j,k]    N M K
        for j in range(0, M):
            denominator = 0;
            for k in range(0, K):
                p[i, j, k] = lamda[i, k] * theta[k, j];
                denominator += p[i, j, k];
            
            if denominator == 0:
                for k in range(0, K):
                    p[i, j, k] = 0;
            else:
                for k in range(0, K):
                    p[i, j, k] /= denominator;

def MStep():
    # update theta
    for k in range(0, K):#修改 theta[k, j]   K M
        denominator = 0
        for j in range(0, M):
            theta[k, j] = 0
            for i in range(0, N):
                theta[k, j] += X[i, j] * p[i, j, k]
            denominator += theta[k, j]
            
        if denominator == 0:
            for j in range(0, M):
                theta[k, j] = 1.0 / M
        else:
            for j in range(0, M):
                theta[k, j] /= denominator
        
    # update lamda
    for i in range(0, N):#修改lamda[i,k]  N K
        for k in range(0, K):
            lamda[i, k] = 0
            denominator = 0
            for j in range(0, M):
                lamda[i, k] += X[i, j] * p[i, j, k]
                denominator += X[i, j];
            
            if denominator == 0:
                lamda[i, k] = 1.0 / K
            else:
                lamda[i, k] /= denominator
# --------------------------------------------------------------------------------
def LogLikelihood():
    loglikelihood = 0
    for i in range(0, N):
        for j in range(0, M):
            tmp = 0
            for k in range(0, K):
                tmp += theta[k, j] * lamda[i, k]
            if tmp > 0:
                loglikelihood += X[i, j] * log(tmp)
    return loglikelihood
# --------------------------------------------------------------------------------
maxIteration = 12
LogLikelihood_list = []
for i in range(0, maxIteration):
    EStep()
    MStep()
    one_LogLikelihood = LogLikelihood()
    LogLikelihood_list.append(one_LogLikelihood)

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print("lamda",lamda)
print("theta",theta)
print("LogLikelihood_list",LogLikelihood_list)
print("")

import matplotlib.pyplot as plt

x_data = np.arange(len(LogLikelihood_list))
y_data = np.asarray(LogLikelihood_list)
plt.plot(x_data,y_data)
plt.show()

# --------------------------------------------------------------------------------
topicWordsNum = 4
def output_topic_words():
    for i in range(0, K):
        topicword = []
        ids = theta[i, :].argsort()
        for j in ids:
            topicword.insert(0, id2word[j])
        tmp = ''
        for word in topicword[0:min(topicWordsNum, len(topicword))]:
            tmp += word + ' '
        print("topic", i, tmp)
output_topic_words()
print("")
