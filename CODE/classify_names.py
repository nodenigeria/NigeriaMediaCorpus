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

	stop_words = ['assistant', 'editor', 'asst.', 'editor', 'reporter', 'reports', 'by', 'from', 'abuja', 'lagos', 'benin city', 'our', 'dr', 'correspondent', 'rev.', 'mr.', 'author', ';']

	with open('names_classifier.pkl', 'rb') as f:
		model, labels = pickle.load(f)

	uncertain_predictions = []
	random_names = []

	for website in websites:
		names = []

		print(website)
		if os.path.splitext(os.path.basename(website))[0] != 'punch_articles':

			with open(website, 'r') as f:
				tsv_reader = csv.reader(f, delimiter="\t")
				for article in tsv_reader:
					name = article[3]
					name = unicodeToAscii(name).lower()
					for word in stop_words:
						name = name.replace(word, ' ')
					name_list = name.split('and')
					names.extend([name.strip() for name in name_list])

				names = list(set(list(names)))
				predictions = model.predict_proba(names)
				class_predictions = model.predict(names)
				
				random_sample = np.random.choice(len(names), 250)
				random_names.extend(zip(np.array(names)[random_sample], np.array(class_predictions)[random_sample]))

				'''
				predictions.sort(axis=1)			
						
				ratios = []
						
				for prediction in predictions:
					ratios.append(prediction[2]/prediction[1])
						
				ratios = sorted(enumerate(ratios), key=lambda x: x[1])
						
				for i in range(0, 100):
					uncertain_predictions.append((names[ratios[i][0]], labels[class_predictions[ratios[i][0]]]))

				'''


						
				'''
				print(os.path.splitext(os.path.basename(website))[0]+'\n')
				print("{}: {} / {} \n".format(labels[0], len([i for i in predictions if i == 0]), len(predictions)))
				print("{}: {} / {} \n".format(labels[1], len([i for i in predictions if i == 1]), len(predictions)))
				print("{}: {} / {} \n".format(labels[2], len([i for i in predictions if i == 2]), len(predictions)))
				'''

	with open('random_names.csv', 'w') as f:	
		random_names_with_class = []
		for name in random_names:
			random_names_with_class.append((name[0], labels[name[1]]))
		writer = csv.writer(f, delimiter=',')
		for prediction in random_names_with_class:
			writer.writerow(prediction)
