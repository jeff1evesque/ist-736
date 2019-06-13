#!/usr/bin/python

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from brain.exploratory.sentiment import Sentiment
from brain.exploratory.word_cloud import word_cloud
from brain.utility.dataframe import cleanse

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
    clean=True
):
    '''

    generate wordclouds and sentiment series plot.

    @target, dataframe column to parse
    @sent_cases, dict where keys represent columns, and values represent list
        of possible column values. This is used to filter the dataframe.

    '''

    if clean:
        df[target] = [str(w).lower() for w in df[target] if w]
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
                wc_temp = df.loc[df[k] == v]

                if not os.path.exists('{d}/{value}'.format(
                    d=directory,
                    value=v
                )):
                    os.makedirs('{d}/{value}'.format(
                        d=directory,
                        value=v
                    ))

                if plot_wc:
                    # wordcloud: using series like dictionary input
                    if (wc_temp[target].size > 1):
                        word_cloud(
                            wc_temp[target].values,
                            filename='{d}/{value}/wc{suffix}.png'.format(
                                d=directory,
                                value=v,
                                suffix=suffix
                            ),
                            stopwords=stopwords,
                            background_color=background_color
                        )

                    # wordcloud: using list of sentences input
                    else:
                        [word_cloud(
                            x,
                            filename='{d}/{value}/wc{suffix}.png'.format(
                                d=directory,
                                value=v,
                                suffix=suffix
                            ),
                            stopwords=stopwords,
                            background_color=background_color
                        ) for x in wc_temp[target]]

                if plot_sentiment:
                    # create sentiment plot
                    sent_temp = Sentiment(wc_temp, column_name=target)
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
            sent_overall = Sentiment(df, column_name=target)
            sent_overall.vader_analysis()
            sent_overall.plot_ts(
                title='Overall Sentiment',
                filename='{d}/sentiment_overall{suffix}.png'.format(
                    d=directory,
                    suffix=suffix
                )
            )

    else:
        # clean text
        if plot_wc:
            # create wordcloud
            word_cloud(
                df[target],
                filename='{d}/wc{suffix}.png'.format(
                    d=directory,
                    suffix=suffix
                ),
                stopwords=stopwords,
                background_color=background_color
            )

        if plot_sentiment:
            # create sentiment plot
            sent_temp = Sentiment(df[target])
            sent_temp.vader_analysis()
            sent_temp.plot_ts(
                title='Sentiment Analysis',
                filename='{d}/sentiment{suffix}.png'.format(
                    d=directory,
                    suffix=suffix
                )
            )
