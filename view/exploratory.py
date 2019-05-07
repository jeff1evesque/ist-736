#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from exploratory.sentiment import Sentiment
from exploratory.word_cloud import word_cloud

def explore(df, sent_cases, target='text'):
    '''

    generate wordclouds and sentiment series plot.

    @target, dataframe column to parse
    @sent_cases, dict where keys represent columns, and
        values represent list of possible column values.

    '''

    cases = []
    for k,val in sent_cases.items():
        for v in val:
            wc_temp = df.loc[df[k] == v]

            # create wordcloud
            word_cloud(
                wc_temp[target],
                filename='viz/{value}/wc_{key}.png'.format(value=v, key=k)
            )

            # create sentiment plot
            sent_temp = Sentiment(wc_temp, target)
            sent_temp.vader_analysis()
            sent_temp.plot_ts(
                title='{value}'.format(value=v),
                filename='viz/{value}/sentiment_{key}.png'.format(value=v, key=k)
            )

        word_cloud(df[target], filename='viz/wc_overall.png')
        sent_overall = Sentiment(df, target)
        sent_overall.vader_analysis()
        sent_overall.plot_ts(
            title='Overall Sentiment',
            filename='viz/sentiment_overall.png'
        )
