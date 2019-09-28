#!/usr/bin/python

import re
from pathlib import Path
from brain.algorithm.text_classifier import Model as alg
from config import model_config as c


def model(
    df=None,
    model_type='multinomial',
    key_class='screen_name',
    max_length=280,
    ngram=(1,1),
    validate=True,
    stopwords=None
):
    '''

    return trained classifier.

    '''

    # initialize classifier
    if df is not None:
        model = alg(
            df=df,
            key_text=cfg['classify_index'],
            key_class=key_class,
            ngram=ngram,
            split_size=c['split_size'],
            stopwords=stopwords
        )
    else:
        model = alg(
            key_text=cfg['classify_index'],
            key_class=key_class,
            ngram=ngram,
            split_size=c['split_size'],
            stopwords=stopwords
        )

    # vectorize data
    model.split(size=split_size)
    params = model.get_split()

    # split validation
    if not validate:
        validate = False

    elif validate == 'full':
        validate = (params['X_train'], params['y_train'])

    else:
        validate = (params['X_test'], params['y_test'])

    # train classifier
    model.train(
        params['X_train'],
        params['y_train'],
        model_type=model_type,
        validate=validate,
        max_length=max_length,
        k=c['classify_chi2']
    )

    return(model)

def model_pos(
    df,
    model_type='multinomial',
    key_class='Sentiment',
    max_length=280,
    stem=False,
    validate=True,
    stopwords=None
):
    '''

    return initialized model using pos.

    '''

    # initialize classifier
    if df is not None:
        model = alg(
            df=df,
            key_text=cfg['classify_index'],
            key_class=key_class,
            stem=False,
            split_size=c['split_size'],
            stopwords=stopwords
        )
    else:
        model = alg(
            key_text=cfg['classify_index'],
            key_class=key_class,
            stem=False,
            split_size=c['split_size'],
            stopwords=stopwords
        )

    #
    # suffix pos: add part of speech suffix to each word.
    #
    df['pos'] = [sent.split() for sent in df[key_text]]
    df['pos'] = [model.get_pos(x) for x in df['pos']]

    #
    # update model: using cleansed data.
    #
    model.set_df(df)
    model.set_key_text('pos')

    # vectorize data
    model.split(size=split_size)
    params = model.get_split()

    # split validation
    if not validate:
        validate = False

    elif validate == 'full':
        validate = (params['X_train'], params['y_train'])

    else:
        validate = (params['X_test'], params['y_test'])

    # train classifier
    model.train(
        params['X_train'],
        params['y_train'],
        model_type=model_type,
        validate=validate,
        max_length=max_length,
        k=c['classify_chi2']
    )

    return(model)
