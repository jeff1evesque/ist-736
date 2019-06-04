#!/usr/bin/python

import math
from model.timeseries import model
from view.timeseries import plot_ts

def timeseries(
    df,
    normalize_key,
    directory='viz',
    flag_arima=True,
    flag_lstm=True,
    plot=True,
    show=False,
    suffix=None
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
            predicted_df = pd.DataFrame({
                'actual': test_actual,
                'predicted': predicted,
                'dates': dates[:len(dates)]
            })
            predicted_df_long = pd.melt(
                predicted_df,
                id_vars=['dates'],
                value_vars=['actual', 'predicted']
            )

            # plot
            plot_ts(
                data=pd.DataFrame({
                    'values': train_actual,
                    'dates': dates[:len(train_actual)]
                }),
                x='values',
                y='dates',
                directory=directory,
                filename='ts_train'
            )

            plot_ts(
                data=predicted_df_long,
                x='value',
                y='dates',
                hue='variable',
                directory=directory,
                filename='ts_test'
            )

            # trend analysis
            decomposed = l.get_decomposed()
            decomposed.plot()
            plot.savefig(
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
            train_predicted = l.get_predict_test()[0]
            test_actual = l.get_data('total', key_to_list='True')[1]
            test_predicted = l.get_predict_test()[1]

            test_predicted_df = pd.DataFrame({
                'actual': test_actual,
                'predicted': test_predicted,
                'dates': dates[:len(dates)]
            })
            test_predicted_df_long = pd.melt(
                predicted_df,
                id_vars=['dates'],
                value_vars=['actual', 'predicted']
            )

            # plot
            plot_ts(
                data=pd.DataFrame({
                    'values': train_actual,
                    'dates': dates[:len(train_actual)]
                }),
                x='values',
                y='dates',
                directory=directory,
                filename='ts_train'
            )

            plot_ts(
                data=test_predicted,
                x='value',
                y='dates',
                hue='variable',
                directory=directory,
                filename='ts_test'
            )

    # return score
    return(model_scores)
