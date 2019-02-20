from sklearn.base import BaseEstimator, TransformerMixin

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
import pickle

from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.sparse import csr_matrix

import stanfordnlp
from collections import Counter

class SampleExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, vars):
        self.vars = vars  # e.g. pass in a column name to extract

    def transform(self, X, y=None):
        return do_something_to(X, self.vars)  # where the actual feature extraction happens

    def fit(self, X, y=None):
        return self  # generally does nothing

class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def average_word_length(self, name):
        """Helper code to compute average word length of a name"""
        return np.mean([len(word) for word in name.split()])

    def transform(self, X, y=None):

        # Ugh.  All this stupid fancy numpy stuff just to get the damned matrix
        # dimensions to line up :(
        output = np.asmatrix(np.array([self.average_word_length(x) for x in X])).T
        print('awl', output.shape)
        return output

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class POSTagExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        self.nlp = stanfordnlp.Pipeline()
        self.pos_to_index = {}

    def fit_transform(self, X, y=None):
        print('fit-transform called')
        #X_ = []
        row = []
        col = []
        val = []
        for r, text in enumerate(X):
            print('parsing sentence %d' % r)
            doc = self.nlp(text)
            pos_counts = Counter()

            for sent in doc.sentences:
                for w in sent.words:
                    pos = w.pos
                    pos_counts[w.pos] += 1

            for pos, count in pos_counts.items():
                
                if pos not in self.pos_to_index:
                    c = len(self.pos_to_index)
                    self.pos_to_index[pos] = c
                else:
                    c = self.pos_to_index[pos]

                row.append(r)
                col.append(c)
                val.append(count)

        # print(row, col, val)
        #print(row)
        #print(col)
        #print('Converting to sparse matrix of %dx%d' % (r, len(self.pos_to_index)))
        output = np.zeros((r+1, len(self.pos_to_index)))
        # WTF IS GOING ON HERE.  It freezes when r>0 ?????
        for i, r in enumerate(row):
            output[r, col[i]] = val[i]
            #print("[%d, %d] = %d" % (r, col[i], val[i]))
        #output = csr_matrix((val, (row, col)), shape=(r, len(self.pos_to_index)))
        #print('Filled the values')
        print('fit_transform', output.shape)
                    
        return output

    def transform(self, X, y=None):
        print('transform called')
        #X_ = []
        row = []
        col = []
        val = []
        for r, text in enumerate(X):
            doc = self.nlp(text)
            pos_counts = Counter()
            X_.append(pos_counts)
            for sent in doc.sentences:
                for w in sent.words:
                    pos = w.pos
                    pos_counts[w.pos] += 1

            for pos, count in pos_counts.items():
                
                if pos in self.pos_to_index:
                    c = self.pos_to_index[pos]
                else:
                    continue
                    
                row.append(r)
                col.append(c)
                val.append(v)
        
        # NOTE: we need to manually set the shape here to account for features
        # that weren't present (assuming fit had been called prior)
        output = csr_matrix((val, (row, col)), shape=(r, len(self.pos_to_inex)))
        print('transform', output.shape)
                    
        return output.toarray()

    
    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        print('fit called alone???')
        return self

    
class TextNormalizationTransform(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        #print(X)
        output = [re.sub('[^A-Za-z]+', ' ', sentence.lower()) for sentence in X]
        print('tnt %d' % len(output))
        return output
    
    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        print(X)        
        return self    


'''
run with 2 arguments:
(1) pidgin data
(2) english data
'''

def main():

    
    pidgin_data = sys.argv[1]
    english_data = sys.argv[2]

    pidgin_str = ''
    english_str = ''
    
    labels = ['pi', 'en']

    max_lines = 20
    pidgin_lines = []
    with open(pidgin_data) as f:
        for line_no, line in enumerate(f):
            pidgin_lines.append(line)
            if line_no >= max_lines:
                break
            
    english_lines = []
    with open(english_data) as f:
        for line_no, line in enumerate(f):
            english_lines.append(line)
            if line_no >= max_lines:
                break


    
    # tokenize each of the texts into sentences
    pidgin_sentences = sent_tokenize(' '.join(pidgin_lines).replace('.', '. '))
    english_sentences = sent_tokenize(' '.join(english_lines).replace('.', '. '))

    # remove all non-alphabetic characters and lowercase eech sentence
    #pidgin_sentences = [re.sub('[^A-Za-z]+', ' ', sentence.lower()) for sentence in pidgin_sentences]
    #english_sentences = [re.sub('[^A-Za-z]+', ' ', sentence.lower()) for sentence in english_sentences]


    training_data = []
    training_targets = []
    
    for sentence in pidgin_sentences[:5]:
        training_data.append(sentence)
        training_targets.append(0)

    for sentence in english_sentences[:5]:
        training_data.append(sentence)
        training_targets.append(1)

    pipe = Pipeline([
        ('feature-extraction', FeatureUnion([
            ('cgrams', Pipeline([
                ('normalize', TextNormalizationTransform()), 
                ('vect', CountVectorizer(ngram_range=(4,4), analyzer='char')),
                ('tfidf', TfidfTransformer(use_idf=False)),
            ])),
            ('word-length', AverageWordLengthExtractor()),
            ('pos-feats', POSTagExtractor())
        ])),
        ('classifier', LogisticRegression(solver='lbfgs'))])

    model = pipe.fit(training_data, training_targets)

    if True:
        return

        
    # Create train/test splits
    X_train, X_test, y_train, y_test = train_test_split(training_data,
                                                        training_targets,
                                                        test_size=0.2,
                                                        random_state=0)
        
    pipe = Pipeline([('vect', CountVectorizer(ngram_range=(4,4), analyzer='char')),
                     ('tfidf', TfidfTransformer(use_idf=False)), ('lrg', LogisticRegression(solver='lbfgs'))])
    model = pipe.fit(X_train, y_train)
    
    with open('full_model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    main()
    
