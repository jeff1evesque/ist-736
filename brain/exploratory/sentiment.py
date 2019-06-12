#!/usr/bin/python

# this file the following packages:
#
#     pip install nltk
#     pip install vaderSentiment
#

from nltk import download
import pandas as pd
import matplotlib.pyplot as plt
from brain.utility.dataframe import cleanse
from nltk.sentiment.vader import SentimentIntensityAnalyzer
download('vader_lexicon')


class Sentiment():
    '''

    use vader to perform sentiment analyis.

    '''

    def __init__(self, data, column_name=None):
        '''

        define class variables.

        '''

        # local variables
        self.data = data

        if column_name:
            self.column_name = column_name
            self.data = data[column_name]

        else:
            self.column_name = None

        self.data = cleanse(self.data)

    def vader_analysis(self):
        '''

        perform sentiment analysis.

        '''

        analyser = SentimentIntensityAnalyzer()
        sid = SentimentIntensityAnalyzer()
        result = {
            'compound': [],
            'negative': [],
            'neutral': [],
            'positive': []
        }

        # sentiment analysis
        for sent in self.data:
            ss = sid.polarity_scores(sent)

            for k in sorted(ss):
                if k == 'compound':
                    result['compound'].append(ss[k])
                elif k == 'neg':
                    result['negative'].append(ss[k])
                elif k == 'neu':
                    result['neutral'].append(ss[k])
                elif k == 'pos':
                    result['positive'].append(ss[k])

        #
        # append results: duplicate dataframe resolves the panda
        #     'SettingWithCopyWarning' error.
        #
        self.data_adjusted = pd.DataFrame({
            'compound': result['compound'],
            'negative': result['negative'],
            'neutral': result['neutral'],
            'positive': result['positive']
        })

        # return scores
        return(self.data_adjusted)

    def plot_ts(
        self,
        title='Sentiment Analysis',
        filename='sentiment.png',
        show=False,
        alpha=0.6
    ):
        '''

        plot sentiment generated from 'vader_analysis'.

        '''

        # generate plot
        plt.figure()
        with pd.plotting.plot_params.use('x_compat', True):
            self.data_adjusted.negative.plot(color='r', legend=True, alpha=alpha)
            self.data_adjusted.positive.plot(color='g', legend=True, alpha=alpha)
            self.data_adjusted.neutral.plot(color='b', legend=True, alpha=alpha)
        plt.title(title)

        # save plot
        plt.savefig(filename)

        if show:
            plt.show()
        else:
            plt.close()
