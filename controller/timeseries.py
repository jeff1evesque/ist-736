#!/usr/bin/python

import math
from model.timeseries import model

def classify(
    df,
    normalize_key,
    kfold=True,
    rotation=90,
    directory='viz',
    flag_arima=True,
    flag_lstm=True,
    plot=True
):
    '''

    implement designated classifiers.

    '''

    # local variables
    model_scores = {}

    # arima: autoregressive integrated moving average
    if flag_arima:
        # initialize
        a = model(
            df=df,
            normalize_key=normalize_key,
            model_type='arima'
        )
        model_scores['arima'] = {
            'mse': a.get_mse(),
            'adf': a.get_adf()
        }

        if plot:
            #
            # @dates, full date range
            # @train_actual, entire train values
            # @test_actual, entire train values
            # @predicted, only predicted values
            #
            dates = a.get_index()
            train_actual = a.get_difference(
                data=a.get_data(key='total', key_to_list='True')[0],
                diff=1
            )
            test_actual = a.get_differences()[0]
            predicted = a.get_differences()[1]

            # trend analysis
            ts = l.get_decomposed()[0]['index']
            trend = l.get_decomposed()[1]['index']
            seasonality = l.get_decomposed()[2]['seasonality']
            residual = l.get_decomposed()[3]['residual']

    # lstm: long short term memory
    if flag_lstm:
        # intialize
        l = model(
            df=df,
            normalize_key=normalize_key,
            model_type='lstm'
        )

        # predict
        l.predict_test()
        model_scores['lstm'] = {
            'mse': l.get_mse(),
            'history': l.get_fit_history()
        }

        if plot:
            train_actual = l.get_data('total', key_to_list='True')[0]
            train_predicted = l.get_predict_test()[0]]
            test_actual = l.get_data('total', key_to_list='True')[1]
            test_predicted = l.get_predict_test()[1]

    # return score
    return(model_scores)
