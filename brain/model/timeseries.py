#!/usr/bin/python

import re
from pathlib import Path
from brain.algorithm.arima import Arima
from brain.algorithm.lstm import Lstm

def model(
    df,
    normalize_key,
    model_type='arima',
    date_index='date',
    epochs=100
):
    '''

    return trained classifier.

    '''

    # initialize classifier
    if model_type == 'arima':
        model = Arima(
            data=df,
            normalize_key=normalize_key,
            date_index=date_index,
            train=False
        )

        # induce stationarity
        result = model.grid_search()

        #
        # train: if model is stationary make prediction using rolling length.
        #
        # @result[0], returns the adf statistic, and pvalue:
        #
        #     - adf statistic, the more negative, the more likely to reject the
        #         null hypothesis (Ho).
        #
        #     - pvalue <= 0.05, reject Ho since data does not have unit root
        #         and is stationary. Otherwise, data has stationary root then
        #         reject the Ho.
        #
        if result[0][1] <= 0.05:
            iterations = len(model.get_data(key=normalize_key)[1])
            model.train(iterations=iterations, order=result[2])
            return(model, result[2])

        else:
            return(False)

    elif model_type == 'lstm':
        model = Lstm(
            data=df,
            normalize_key=normalize_key,
            date_index=date_index,
        )
        model.train(epochs=epochs)

        return(model)
