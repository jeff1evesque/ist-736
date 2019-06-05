#!/usr/bin/python

import re
from pathlib import Path
from algorithm.arima import Arima
from algorithm.lstm import Lstm

def model(
    df,
    normalize_key,
    model_type='arima',
    date_index='date'
):
    '''

    return trained classifier.

    '''

    # initialize classifier
    if model_type == 'arima':
        model = Arima(
            data=df,
            normalize_key=normalize_key,
            date_index=date_index
        )

        # train: use rolling length
        iterations = len(model.get_data(key=normalize_key)[1])
        model.train(iterations)

    elif model_type == 'lstm':
        model = Lstm(
            data=df,
            normalize_key=normalize_key,
            date_index=date_index
        )
        model.train()

    return(model)
