#!/usr/bin/python

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from exploratory.sentiment import Sentiment
from exploratory.word_cloud import word_cloud
from utility.dataframe import cleanse

def explore(
    df,
    sent_cases=None,
    stopwords=[],
    target='full_text',
    background_color='white',
    directory='viz',
    suffix='',
    plot_wc=True,
    plot_sentiment=True,
    plot_wc_overall=True,
    plot_sentiment_overall=True,
    cleanse=True
):
    '''

    generate wordclouds and sentiment series plot.

    @target, dataframe column to parse
    @sent_cases, dict where keys represent columns, and values represent list
        of possible column values. This is used to filter the dataframe.

    '''

    if cleanse:
        df[target] = cleanse(df, target, ascii=True)
    else:
        df[target] = [re.sub(
            "'",
            '',
            str(s)
        ) for s in df[target]]

    if sent_cases:
        for k,val in sent_cases.items():
            for v in val:
                if not os.path.exists('{d}/{value}'.format(
                    d=directory,
                    value=v
                )):
                    os.makedirs('{d}/{value}'.format(
                        d=directory,
                        value=v
                    )

                if plot_wc:
                    wc_temp = df.loc[df[k] == v]

                    # create wordcloud
                    word_cloud(
                        wc_temp,
                        filename='{d}/{value}/wc{suffix}.png'.format(
                            d=directory,
                            value=v,
                            suffix=suffix
                        ),
                        stopwords=stopwords,
                        background_color=background_color
                    )

                if plot_sentiment:
                    # create sentiment plot
                    sent_temp = Sentiment(wc_temp, target)
                    sent_temp.vader_analysis()
                    sent_temp.plot_ts(
                        title='{value}'.format(value=v),
                        filename='{d}/{value}/sentiment{suffix}.png'.format(
                            d=directory,
                            value=v,
                            suffix=suffix
                        )
                    )

        if plot_wc_overall:
            word_cloud(
                df[target],
                filename='{d}/wc_overall{suffix}.png'.format(
                    d=directory,
                    suffix=suffix
                ),
                stopwords=stopwords,
                background_color=background_color
            )

        if plot_sentiment_overall:
            sent_overall = Sentiment(df, target)
            sent_overall.vader_analysis()
            sent_overall.plot_ts(
                title='Overall Sentiment',
                filename='{d}/sentiment_overall{suffix}.png'.format(
                    d=directory,
                    suffix=suffix
                )
            )

    else:
        if plot_wc:
            # clean text
            wc_temp = df
            wc_temp[target] = [' '.join(x) for x in wc_temp[target]]
            wc_temp[target] = cleanse(df, target, ascii=True)

            # create wordcloud
            word_cloud(
                wc_temp,
                filename='{d}/wc{suffix}.png'.format(
                    d=directory,
                    suffix=suffix
                ),
                stopwords=stopwords,
                background_color=background_color
            )

        if plot_sentiment:
            # create sentiment plot
            sent_temp = Sentiment(wc_temp, target)
            sent_temp.vader_analysis()
            sent_temp.plot_ts(
                title='Sentiment Analysis',
                filename='{d}/sentiment{suffix}.png'.format(
                    d=directory,
                    suffix=suffix
                )
            )
