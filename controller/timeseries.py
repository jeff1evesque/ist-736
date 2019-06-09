#!/usr/bin/python

import math
import pandas as pd
from model.timeseries import model
from view.timeseries import plot_ts
import matplotlib.pyplot as plt

def timeseries(
    df,
    normalize_key,
    directory='viz',
    flag_arima=True,
    flag_lstm=True,
    plot=True,
    show=False,
    suffix=None,
    date_index='date',
    diff=1,
    xticks=True
):
    '''

    implement designated classifiers.

    '''

    # local variables
    model_scores = {}

    if suffix:
        suffix = '_{suffix}'.format(suffix=suffix)
    else:
        suffix=''

    # arima: autoregressive integrated moving average
    if flag_arima:
        # initialize
        a = model(
            df=df,
            normalize_key=normalize_key,
            model_type='arima',
            date_index=date_index
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
            dates = a.get_data()

            if diff > 1:
                train_actual = a.get_difference(
                    data=a.get_data(key=normalize_key, key_to_list='True')[0],
                    diff=diff
                )

            else:
                train_actual = a.get_data(key=normalize_key, key_to_list='True')[0]

            test_actual = a.get_differences()[0]
            predicted = a.get_differences()[1]

            test_predicted_df = pd.DataFrame({
                'actual': test_actual,
                'predicted': predicted,
                'dates': dates[1][date_index][:len(test_actual)]
            })
            test_predicted_df_long = pd.melt(
                test_predicted_df,
                id_vars=['dates'],
                value_vars=['actual', 'predicted']
            )

            # plot
            plot_ts(
                data=pd.DataFrame({
                    'values': train_actual,
                    'dates': dates[0][date_index][:len(train_actual)]
                }),
                xlab='dates',
                ylab='values',
                directory=directory,
                filename='ts_train_arima{s}'.format(s=suffix),
                rotation=90,
                xticks=xticks
            )

            plot_ts(
                data=test_predicted_df_long,
                xlab='dates',
                ylab='value',
                hue='variable',
                directory=directory,
                filename='ts_test_arima{s}'.format(s=suffix),
                rotation=90,
                xticks=xticks
            )

            # trend analysis
            decomposed = a.get_decomposed()
            decomposed.plot()
            plt.savefig(
                '{d}/{f}'.format(
                    d=directory,
                    f='trend{suffix}'.format(suffix=suffix)
                )
            )

            if show:
                plt.show()
            else:
                plt.close()

    # lstm: long short term memory
    if flag_lstm:
        # intialize
        l = model(
            df=df,
            normalize_key=normalize_key,
            model_type='lstm',
            date_index=date_index
        )

        # predict
        l.predict()
        model_scores['lstm'] = {
            'mse': l.get_mse(),
            'history': l.get_fit_history()
        }

        if plot:
            #
            # @dates, full date range
            # @train_actual, entire train values
            # @test_actual, entire train values
            # @predicted, only predicted values
            #
            dates = l.get_data()
            train_actual = l.get_data(normalize_key, key_to_list='True')[0]
            train_predicted = [x[0] for x in l.get_predict_test()[0]]
            test_actual = l.get_data(normalize_key, key_to_list='True')[1]
            test_predicted = [x[0] for x in l.get_predict_test()[1]]

            test_predicted_df = pd.DataFrame({
                'actual': test_actual[-len(test_predicted):],
                'predicted': test_predicted,
                'dates': dates[1][date_index][-len(test_predicted):]
            })
            test_predicted_df_long = pd.melt(
                test_predicted_df,
                id_vars=['dates'],
                value_vars=['actual', 'predicted']
            )

            # plot
            plot_ts(
                data=pd.DataFrame({
                    'values': train_actual,
                    'dates': dates[0][date_index][:len(train_actual)]
                }),
                xlab='dates',
                ylab='values',
                directory=directory,
                filename='ts_train_lstm{s}'.format(s=suffix),
                rotation=90,
                xticks=xticks
            )

            plot_ts(
                data=test_predicted_df_long,
                xlab='dates',
                ylab='value',
                hue='variable',
                directory=directory,
                filename='ts_test_lstm{s}'.format(s=suffix),
                rotation=90,
                xticks=xticks
            )

    # return score
    return(model_scores)
