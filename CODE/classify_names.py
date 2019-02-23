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
	websites = glob.glob(article_directory+'/*.tsv')

	with open('names_classifier.pkl', 'rb') as f:
		model, labels = pickle.load(f)

	uncertain_predictions = []

	for website in websites:
		names = []

		if os.path.splitext(os.path.basename(website)) != 'thepunch':

			with open(website, 'r') as f:
				tsv_reader = csv.reader(f, delimiter="\t")
				for article in tsv_reader:
					name = article[3]
					name = unicodeToAscii(name).lower()
					last_name = name.split(' ')[-1]
					if last_name != 'none' and last_name != 'reporter' and last_name != 'report' and last_name != 'group':
						names.append(last_name)

						predictions = model.predict_proba(names)
						predictions_sorted = np.array(predictions).sort(axis=1)
						
						
						ratios = []
						
						for prediction in predictions_sorted:
							ratios.append(prediction[2]/prediction[1])
						
						ratios = sorted(enumerate(ratios), key=lambda x: x[1])
						
						for i in range(0, 100):
							uncertain_predictions.append(names[ratios[i][0]])



						
						'''
						print(os.path.splitext(os.path.basename(website))[0]+'\n')
						print("{}: {} / {} \n".format(labels[0], len([i for i in predictions if i == 0]), len(predictions)))
						print("{}: {} / {} \n".format(labels[1], len([i for i in predictions if i == 1]), len(predictions)))
						print("{}: {} / {} \n".format(labels[2], len([i for i in predictions if i == 2]), len(predictions)))
						'''

	with open('uncertain_predictions.txt', 'w') as f:
		for prediction in uncertain_predictions:
			f.write(prediction)
