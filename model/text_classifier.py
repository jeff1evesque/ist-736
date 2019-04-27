#!/usr/bin/python

#
# this files the following packages:
#
#     pip install nltk
#     pip install -U scikit-learn
#     pip install scikit-plot
#

import time
import csv
import re
import string
import numpy as np
from pathlib import Path
import pandas as pd
from nltk.corpus import stopwords
from nltk import tokenize, download, pos_tag
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import cross_val_predict, train_test_split
import matplotlib.pyplot as plt
import scikitplot as skplt
from model.penn_treebank import penn_scale
stop_words = set(stopwords.words('english'))
download('vader_lexicon')


class Model():
    '''

    train classifier model using provided dataset.

    '''

    def __init__(
        self,
        df=None,
        vectorize=True,
        key_text='SentimentText',
        key_class='Sentiment',
        stem=True,
        lowercase=True,
        fp='{}/data/sample-sentiment.csv'.format(
            Path(__file__).resolve().parents[1]
        )
    ):
        '''

        define class variables.

        '''

        # class variables
        self.key_text = key_text
        self.key_class = key_class
        self.actual = None
        self.predicted = None

        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(fp)

        #
        # clean: remove twitter account, punctuations and urls, lowercase,
        #        stem each word.
        #
        # @string.punctuation, '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        #
        pattern_twitter_act = '@[a-zA-Z0-9_]{0,15}'
        pattern_url = 'https?://[A-Za-z0-9./]+'
        pattern_punctuation = '[{p}]'.format(p=string.punctuation)
        pattern = '|'.join((pattern_twitter_act, pattern_url, pattern_punctuation))

        self.df[self.key_text] = [re.sub(pattern, '', w) for w in self.df[self.key_text]]

        if lowercase:
            self.df[self.key_text] = [w.lower() for w in self.df[self.key_text]]

        if stem:
            porter = PorterStemmer()
            self.df[self.key_text] = [porter.stem(w) for w in self.df[self.key_text]]

        # vectorize data
        if vectorize:
            self.split()
            self.vectorize()

    def split(self, size=0.20, pos_split=False):
        '''

        split data into train and test.

        '''

        # split
        if pos_split:
            for i, row in self.df.iterrows():
                # max length
                if isinstance(self.df[self.key_text].iloc[i], str):
                    max_length = len(self.df[self.key_text].iloc[i].split())
                else:
                    max_length = len(self.df[self.key_text].iloc[i].str.split())
                pos = self.df[['pos']].iloc[i]

                # rebuild 'key-text' with pos suffix
                combined = ''
                for j in range(max_length):
                    if isinstance(self.df[self.key_text][i], str):
                        word = self.df[self.key_text][i].split()[j]
                    else:
                        word = self.df[self.key_text].iloc[i].split()[j]

                    combined = '{combined} {word}-{pos}'.format(
                        combined=combined,
                        word=word,
                        pos=pos[0][j]
                    )
                self.df[self.key_text].iloc[[i]] = combined

        # split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df[self.key_text],
            self.df[self.key_class],
            test_size=size
        )

    def get_split(self):
        '''

        return previous train and test split.

        '''

        return({
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
        })

    def get_pos(self, l):
        '''

        apply pos tagger to supplied list.

        '''

        pos = pos_tag(l)
        result = ' '.join(['{word}-{pos}'.format(
            word=l[i],
            pos=penn_scale[x[1]]
        ) if x[1] in penn_scale else '{word}-{pos}'.format(
            word=l[i],
            pos=1
        ) for i,x in enumerate(pos)])
        return(result)

    def vectorize(self, stop_words='english'):
        '''

        vectorize provided data.

        '''

        # bag of words: with 'english' stopwords
        self.count_vect = CountVectorizer(stop_words=stop_words)
        self.bow = self.count_vect.fit_transform(self.X_train)

        # tfidf weighting
        self.tfidf_transformer = TfidfTransformer()
        self.X_train_tfidf = self.tfidf_transformer.fit_transform(self.bow)

    def get_tfidf(self):
        '''

        get current X_train tfidf.

        '''

        return(self.X_train_tfidf)

    def get_df(self):
        '''

        get original dataframe.

        '''

        return(self.df)

    def model(self, X, y, validate=False, max_length=280, model_choice=None, multiclass=False):
        '''

        create classifier model.

        @validate, must have tuple shape (X_test, y_test)
        @model_choice, override default behavior with defined model.

            - bernoulli
            - multinomial (default)
            - svm, with linear kernel since text is high dimensional.

        @multiclass, svm indicator of greater than binary classification.

        '''

        # conditionally select model
        if (model_choice == 'svm'):
            if multiclass:
                clf = svm.SVC(gamma='scale', kernel='linear', decision_function_shape='ovo')
            else:
                clf = svm.SVC(gamma='scale', kernel='linear')

            tfidf_transformer = TfidfTransformer()
            data = self.tfidf_transformer.fit_transform(bow)

        elif (
            (model_choice == 'bernoulli') or
            (not model_choice and all(len(sent) <= max_length for sent in self.X_train))
        ):
            self.clf = BernoulliNB()
            self.clf.fit(X, y)

            # validate
            if validate and len(validate) == 2:
                predictions = []

                for item in list(validate[0]):
                    prediction = self.count_vect.transform([item])
                    predictions.append(
                        self.clf.predict(prediction)
                    )

                self.actual = validate[1]
                self.predicted = predictions

                return({
                    'model': self.clf,
                    'actual': validate[1],
                    'predicted': predictions
                })

            return({
                'model': self.clf,
                'actual': None,
                'predicted': None
            })

        else:
            self.clf = MultinomialNB()
            self.clf.fit(X, y)

            # validate
            if validate and len(validate) == 2:
                predictions = []
                for item in list(validate[0]):
                    prediction = self.count_vect.transform([item])
                    predictions.append(
                        self.clf.predict(self.tfidf_transformer.fit_transform(prediction))
                    )

                self.actual = validate[1]
                self.predicted = predictions

                return({
                    'model': self.clf,
                    'actual': validate[1],
                    'predicted': predictions
                })

            return({
                'model': self.clf,
                'actual': None,
                'predicted': None
            })

    def plot_cm(
        self,
        actual=None,
        predicted=None,
        filename='confusion_matrix.png'
    ):
        '''

        plot sentiment generated from 'vader_analysis'.

        '''

        if not actual:
            actual = self.actual
        if not predicted:
            predicted = self.predicted

        # generate plot
        plt.figure()
        skplt.metrics.plot_confusion_matrix(actual, predicted)

        # save plot
        plt.savefig(filename)
        plt.show()

    def get_accuracy(self, actual=None, predicted=None):
        '''

        return accuracy for prediction.

        '''

        if not actual:
            actual = self.actual
        if not predicted:
            predicted = self.predicted

        return(accuracy_score(actual, predicted))


    def get_kfold_scores(
        self,
        kfold,
        stop_words='english',
        max_length=280,
        shuffle=True,
        model_choice=None,
        multiclass=False
    ):
        '''

        return kfold validation scores. Variance between scores is
        an indication that either the algorithm is unable to learn,
        or the data may require additional cleaning and preprocessing.

        @model_choice, override default behavior with defined model.

            - bernoulli
            - multinomial (default)
            - svm, with linear kernel since text is high dimensional.

        @multiclass, svm indicator of greater than binary classification.

        '''

        kf = KFold(self.df, n_folds=size, shuffle=shuffle)

        # bag of words: with 'english' stopwords
        count_vect = CountVectorizer(stop_words=stop_words)
        bow = self.count_vect.fit_transform(self.df[self.key_text])

        # conditionally select model
        if (model_choice == 'svm'):
            if multiclass:
                clf = svm.SVC(gamma='scale', kernel='linear', decision_function_shape='ovo')
            else:
                clf = svm.SVC(gamma='scale', kernel='linear')

            tfidf_transformer = TfidfTransformer()
            data = self.tfidf_transformer.fit_transform(bow)

        elif (
            (model_choice == 'bernoulli') or
            (not model_choice and all(len(sent) <= max_length for sent in self.X_train))
        ):
            clf = BernoulliNB()
            data = bow

        else:
            clf = MultinomialNB()
            tfidf_transformer = TfidfTransformer()
            data = self.tfidf_transformer.fit_transform(bow)

        return(
            cross_val_predict(
                clf,
                data,
                y=self.df[self.key_class],
                cv=kfold
            )
        )
