import os
import glob
import sys

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import numpy as np
import scipy

from nltk.tokenize import sent_tokenize

import pandas as pd
import pickle

'''
run with 1 argument:
(1) directory containing comment json
'''

if __name__ == "__main__":

	comments_directory = sys.argv[1]



	# remove all non-alphabetic characters and lowercase eech sentence
	pidgin_sentences = [re.sub('[^A-Za-z]+', ' ', sentence.lower()) for sentence in pidgin_sentences]
	english_sentences = [re.sub('[^A-Za-z]+', ' ', sentence.lower()) for sentence in english_sentences]


	test_data = []

	for file in glob.glob(directory+"*.json"):
		with open(file, 'r') as f:
			comment_data = json.loads(file)
			for uri, comments in comment_data:
				if comments:
					comment_string = comments["message"]["value"]
					comment_sentences = sent_tokenize(comment_str.replace('.', '. '))
					comment_sentences = [re.sub('[^A-Za-z]+', ' ', sentence.lower()) for sentence in comment_sentences]
					for sentence in comment_sentences:
						test_data.append(sentence)

	with open('logistic_model.pkl', 'rb') as f:
		model = pickle.load(f)

	y_predicted = model.decision_function(test_data)

	sorted_predictions = sorted(enumerate(y_predicted), key = lambda x: x[1])

	lowest_prediced = sorted_predictions[0:100]
	highest_predicted = sorted_predictions[-100:]

	for sentence in lowest_prediced:
		print(sentence)

	for sentence in highest_predicted:
		print(sentence)