#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 15:53:49 2020
Natural Language Processing
@author: jacob
"""

import nltk
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns

# Preprocess
from sklearn.preprocessing import LabelEncoder

import string
from nltk.corpus import stopwords

# Vectorizing words

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# for keras models

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#nltk.download_shell()
sms_data = pd.read_csv('smsspamcollection/SMSSpamCollection', delimiter = '\t', header= None, names = ['target', 'message'])
#sms_data.columns = ['target', 'message']

#Get an idea of the data
sms_data.groupby(by = 'target').describe()

# Plot messages and length of messages 

sms_data['length'] = sms_data.message.apply(len)

sms_data.length.plot.hist(bins = 100)
sms_data.hist(column = 'length', by = 'target', bins = 100)
sms_data.length.describe()



from sklearn.model_selection import train_test_split
X = sms_data.iloc[:,1:]
y = sms_data.iloc[:,0]

labelEncoder_Y = LabelEncoder()
y = labelEncoder_Y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)



# Bag of words
# 

def text_process(mess):
# =============================================================================
# Remove punctuation
# remove stop words
# return list of cleaned words
# =============================================================================
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    


#sms_data['vector'] = sms_data.message.apply(text_process)
bow_transformer = CountVectorizer(analyzer=text_process)
messages_bow = bow_transformer.fit_transform(X_train['message'])
tfidf_transformer = TfidfTransformer()
messages_tfidf = tfidf_transformer.fit_transform(messages_bow)

tfidf_transformer.transform(bow_transformer.transform(X_test['message']))

model = Sequential()
model.add(Dense(512,kernel_initializer = 'uniform', activation = 'relu', input_dim=messages_tfidf.shape[1]))
# model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(512,kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))


model.compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

history = model.fit(messages_tfidf, y_train, epochs=4, batch_size=32,verbose=1)


history.model.predict(tfidf_transformer.transform(bow_transformer.transform(X_test['message'])))

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
confusion_matrix(y,history.model.predict(messages_tfidf)>0.5)
ann_prediction = history.model.predict(tfidf_transformer.transform(bow_transformer.transform(X_test['message'])))
ann_prediction_class = ann_prediction > 0.5
confusion_matrix(y_test,ann_prediction_class)


roc = roc_auc_score(y_train, model.predict(messages_tfidf))

from sklearn.linear_model import LogisticRegression


classifier = LogisticRegression()
classifier.fit(messages_tfidf,y_train)
logistic_prediction = classifier.predict(tfidf_transformer.transform(bow_transformer.transform(X_test['message'])))
confusion_matrix(y_test, logistic_prediction)


# X_train_sparse = sparse.csr_matrix(messages_tfidf)
# samples_per_epoch=X_train_sparse.shape[0]

# def batch_generator(X, y, batch_size):
#     number_of_batches = samples_per_epoch/batch_size
#     counter=0
#     shuffle_index = np.arange(np.shape(y)[0])
#     np.random.shuffle(shuffle_index)
#     X =  X[shuffle_index, :]
#     y =  y[shuffle_index]
#     while 1:
#         index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
#         X_batch = X[index_batch,:].todense()
#         y_batch = y[index_batch]
#         counter += 1
#         yield(np.array(X_batch),y_batch)
#         if (counter < number_of_batches):
#             np.random.shuffle(shuffle_index)
#             counter=0
            
# model.fit_generator(generator=batch_generator(X_train_sparse, Y_train, batch_size),
#                     nb_epoch=nb_epoch, 
#                     samples_per_epoch=X_train_sparse.shape[0])
