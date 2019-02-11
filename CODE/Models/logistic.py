import os
import glob
import sys

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

import numpy as np
import scipy

from nltk.tokenize import sent_tokenize

import pandas as pd

'''
run with 2 arguments:
(1) pidgin data
(2) english data
'''

if __name__ == "__main__":
	pidgin_data = sys.argv[1]
	english_data = sys.argv[2]

	pidgin_str = ''
	english_str = ''

	labels = ['pi', 'en']

	with open(pidgin_data) as f:
		pidgin_str = f.read().replace('\n', '').decode('utf-8')

	with open(english_data) as f:
		english_str = f.readline().decode('utf-8')


	# tokenize each of the texts into sentences
	pidgin_sentences = sent_tokenize(pidgin_str.replace('.', '. '))
	english_sentences = sent_tokenize(english_str.replace('.', '. '))



	# remove all non-alphabetic characters and lowercase eech sentence
	pidgin_sentences = [re.sub('[^A-Za-z]+', ' ', sentence.lower()) for sentence in pidgin_sentences]
	english_sentences = [re.sub('[^A-Za-z]+', ' ', sentence.lower()) for sentence in english_sentences]


	training_data = []
	training_targets = []

	for sentence in pidgin_sentences:
		training_data.append(sentence)
		training_targets.append(0)

	for sentence in english_sentences:
		training_data.append(sentence)
		training_targets.append(1)

	# Create train/test splits
	X_train, X_test, y_train, y_test = train_test_split(training_data,
                                                                     training_targets,
                                                                     test_size=0.2,
                                                                     random_state=0)

	'''
	bigram_pipe = Pipeline([('vect', CountVectorizer(ngram_range=(2,2), analyzer='word')), 
		('tfidf', TfidfTransformer(use_idf=False)), ('lrg', LogisticRegression(solver='lbfgs'))])

	bigram_model = bigram_pipe.fit(X_train, y_train)

	sixgram_pipe = Pipeline([('vect', CountVectorizer(ngram_range=(1,1), analyzer='char')), 
		('tfidf', TfidfTransformer(use_idf=False)), ('lrg', LogisticRegression(solver='lbfgs'))])

	sixgram_model = sixgram_pipe.fit(X_train, y_train)

	y_predicted_bigram = bigram_model.predict(X_test)
	y_predicted_sixgram = sixgram_model.predict(X_test)

	print(accuracy_score(y_test, y_predicted_bigram))
	print(accuracy_score(y_test, y_predicted_sixgram))
	'''
	

	# create classifier for each word n-gram for n in range [1, 4)
	for i in range(1, 4):
		pipe = Pipeline([('vect', CountVectorizer(ngram_range=(i, i), analyzer='word')),
			('tfidf', TfidfTransformer(use_idf=False)), ('lrg', LogisticRegression(solver='lbfgs'))])
		model = pipe.fit(X_train, y_train)
		y_predicted = model.predict(X_test)
		print("{} word gram auroc: {}".format(i, roc_auc_score(y_test, y_predicted)))

	# create classifier for each character n-gram for n in range [3, 7)
	for i in range(3, 7):
		pipe = Pipeline([('vect', CountVectorizer(ngram_range=(i,i), analyzer='char')),
			('tfidf', TfidfTransformer(use_idf=False)), ('lrg', LogisticRegression(solver='lbfgs'))])
		model = pipe.fit(X_train, y_train)
		y_predicted = model.predict(X_test)
		print("{} char gram auroc: {}".format(i, roc_auc_score(y_test, y_predicted)))
