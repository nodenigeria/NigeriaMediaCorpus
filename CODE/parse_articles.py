import os
import glob
import json
import sys
import csv

'''
run with 2 arguments:
(1) path to article data
(2) path to save JSON result
'''

csv.field_size_limit(sys.maxsize)

if __name__ == "__main__":
	article_data = sys.argv[1]
	path_to_save = sys.argv[2]

	result = {}

	# creates object for each article indexed by URI

	with open(article_data) as f:
		tsv_reader = csv.reader(f, delimiter="\t")
		for article in tsv_reader:
			result[article[6]] = dict(datetime=article[0], section=article[1], title=article[2], 
				author=article[3][3:], text=article[4], source=article[5])

	with open(path_to_save, "w") as f:
		encodingString = json.dumps(result, indent=4)
		f.write(encodingString)
