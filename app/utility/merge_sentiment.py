#!/usr/bin/python

import pandas as pd
from brain.exploratory.sentiment import Sentiment


def merge_sentiment(df, source_col, df_index='created_at'):
    '''

    apply vader sentiment analysis on supplied dataframe column, then append
    corresponding scores to the same dataframe.

    '''

    s = Sentiment(df, source_col)
    df = pd.concat(
        [df, s.vader_analysis()],
        axis=1,
        sort=False,
        verify_integrity=True
    )

    df.replace('\s+', ' ', regex=True, inplace=True)
    df.set_index(df_index, inplace=True)
    df.index.name = df_index

    return(df)
