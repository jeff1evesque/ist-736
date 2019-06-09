#!/usr/bin/python

import numpy as np
import pandas as pd
from model.topic_model import model_lda, model_nmf
from view.exploratory import explore

def topic_model(
    df,
    max_df=1.0,
    min_df=1,
    random_state=None,
    alpha=.1,
    l1_ratio=.5,
    init='nndsvd',
    num_words=40,
    num_topics=10,
    max_iter=5,
    max_features=500,
    learning_method='online',
    learning_offset=50.,
    directory='viz',
    rotation=90,
    flag_lda=True,
    flag_nmf=True,
    plot=True,
    plot_sentiment_overall=True,
    vectorize_stopwords='english',
    stopwords=[],
    auto=False,
    ngram=1
):
    '''

    implement topic model.

    '''

    if ngram > 1:
        suffix = '_ngram'
    else:
        suffix = ''

    if flag_lda:
        lda = model_lda(
            df,
            key_text='text',
            max_df=max_df,
            min_df=min_df,
            num_topics=num_topics,
            max_iter=max_iter,
            max_features=max_features,
            learning_method=learning_method,
            learning_offset=learning_offset,
            random_state=random_state,
            vectorize_stopwords=vectorize_stopwords,
            stopwords=stopwords,
            auto=auto,
            ngram=ngram
        )
        lda_words = lda.get_topic_words(
            feature_names=lda.get_feature_names(),
            num_words=num_words
        )

        if plot:
            explore(
                df=pd.DataFrame(
                    lda_words,
                    columns=['topics', 'words']
                ),
                target='words',
                suffix='_lda{suffix}'.format(suffix=suffix),
                sent_cases={'topics': [x[0] for x in lda_words]},
                plot_sentiment=False,
                plot_sentiment_overall=plot_sentiment_overall,
                cleanse=False
            )

    if flag_nmf:
        nmf = model_nmf(
            df,
            key_text='text',
            max_df=max_df,
            min_df=min_df,
            num_topics=num_topics,
            random_state=random_state,
            max_features=max_features,
            alpha=alpha,
            l1_ratio=l1_ratio,
            init=init,
            stopwords=stopwords,
            vectorize_stopwords=vectorize_stopwords,
            auto=auto,
            ngram=ngram
        )
        nmf_words = nmf.get_topic_words(
            feature_names=nmf.get_feature_names(),
            num_words=num_words
        )

        if plot:
            explore(
                df=pd.DataFrame(
                    nmf_words,
                    columns=['topics', 'words']
                ),
                target='words',
                suffix='_nmf{suffix}'.format(suffix=suffix),
                sent_cases={'topics': [x[0] for x in nmf_words]},
                plot_sentiment=False,
                plot_sentiment_overall=plot_sentiment_overall,
                cleanse=False
            )
