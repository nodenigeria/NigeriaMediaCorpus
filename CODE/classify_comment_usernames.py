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
(1) directory containing comment ddata
'''

if __name__ == "__main__":
	comment_directory = sys.argv[1]
	websites = glob.glob(comment_directory+'/*.json')

	with open('names_classifier.pkl', 'rb') as f:
		model, labels = pickle.load(f)

	random_names = []

	for website in websites:
		names = []

		with open(website, 'r') as f:
			comments = json.load(f)

			for article in comments:

				if article:
					for post in article
						name = post["author"]["value"]
						name = unicodeToAscii(name).lower()
						names.append(name)

			names = list(set(list(names)))
			predictions = model.predict_proba(names)
			class_predictions = model.predict(names)
			
			random_sample = np.random.choice(len(names), 250)
			random_names.extend(zip(np.array(names)[random_sample], np.array(class_predictions)[random_sample]))


	with open('random_usernames.csv', 'w') as f:	
		random_names_with_class = []
		for name in random_names:
			random_names_with_class.append((name[0], labels[name[1]]))
		writer = csv.writer(f, delimiter=',')
		for prediction in random_names_with_class:
			writer.writerow(prediction)
