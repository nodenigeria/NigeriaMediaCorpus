import pickle
import os
import glob
import sys

import re
import numpy as np
import pandas as pd

import csv
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

'''
run with 1 argument:
(1) directory containing article data
'''

if __name__ == "__main__":
	article_directory = sys.argv[1]
	websites = glob.glob(directory+'/*.tsv')

	with open('names_classifier.pkl', 'rb') as f:
		vectorizer, model, labels = pickle.load(f)

	for website in websites:
		names = []
		with open(website, 'r') as f:
			tsv_reader = csv.reader(f, delimiter="\t")
			for article in tsv_reader:
				names.append(unicodedata(article[3].split(' ')[-1:].lower()))

		X = vectorizer.fit_transform(names)
		predictions = model.predict(X)

		print(os.path.splitext(os.path.basename(website))[0]+'\n')
		print("{}: {} / {} \n".format(labels[0], len([i for i in predictions if i == 0]), len(predictions)))
		print("{}: {} / {} \n".format(labels[1], len([i for i in predictions if i == 1]), len(predictions)))
		print("{}: {} / {} \n".format(labels[2], len([i for i in predictions if i == 2]), len(predictions)))

