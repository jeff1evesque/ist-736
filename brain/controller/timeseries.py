#!/usr/bin/python

import math
import numpy as np
import pandas as pd
from brain.model.timeseries import model
from brain.view.timeseries import plot_ts
import matplotlib.pyplot as plt

class Timeseries():
    '''

    controller for arima and lstm models.

    '''

    def __init__(
        self,
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
        xticks=True,
        lstm_epochs=100,
        lstm_dropout=0.2,
        lstm_batch_size=32,
        lstm_validation_split=0,
        lstm_activation='linear'
        auto_scale=False,
        rolling_grid_search=False,
        catch_grid_search=False
    ):

        # local variables
        self.df = df
        self.model_scores = {}

        if suffix:
            suffix = '_{suffix}'.format(suffix=suffix)
        else:
            suffix=''

        # implement models
        if flag_arima:
            self.arima(
                normalize_key=normalize_key,
                log_transform=0.01,
                date_index=date_index,
                auto_scale=auto_scale,
                rolling_grid_search=rolling_grid_search,
                catch_grid_search=catch_grid_search,
                directory=directory,
                suffix=suffix
            )

        if flag_lstm:
            self.lstm(
                normalize_key=normalize_key,
                date_index=date_index,
                epochs=lstm_epochs,
                dropout=lstm_dropout,
                batch_size=lstm_batch_size,
                validation_split=lstm_validation_split,
                activation='linear',
                directory=directory,
                suffix=suffix
            )

    def arima(
        self,
        normalize_key,
        directory='viz',
        plot=True,
        show=False,
        suffix=None,
        date_index='date',
        diff=1,
        xticks=True,
        auto_scale=False,
        log_transform=0.01,
        rolling_grid_search=False,
        catch_grid_search=False
    ):
        '''

        implement arima model.

        '''

        # initialize
        a = model(
            df=self.df,
            normalize_key=normalize_key,
            log_transform=log_transform,
            model_type='arima',
            date_index=date_index,
            auto_scale=auto_scale,
            rolling_grid_search=rolling_grid_search,
            catch_grid_search=catch_grid_search
        )

        if a and isinstance(a, (tuple, list, set)) and len(a) == 2:
            arima_suffix = '{s}_{order}'.format(
                s=suffix,
                order='-'.join([str(x) for x in a[1]])
            )

            self.model_scores['arima'] = {
                'mse': a[0].get_mse(),
                'adf': a[0].get_adf()
            }

            if plot:
                #
                # @dates, full date range
                # @train_actual, entire train values
                # @test_actual, entire train values
                # @predicted, only predicted values
                #
                dates = a[0].get_data()
                if diff > 1:
                    train_actual = a[0].get_difference(
                        data=a[0].get_data()[0],
                        diff=diff
                    )

                else:
                    train_actual = a[0].get_data()[0]

                test_actual = a[0].get_differences()[0]
                predicted = a[0].get_differences()[1]

                test_predicted_df = pd.DataFrame({
                    'actual': test_actual,
                    'predicted': predicted,
                    'dates': a[0].get_data()[1].index
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
                        'dates': dates[0].index.values
                    }),
                    xlab='dates',
                    ylab='values',
                    directory=directory,
                    filename='ts_train_arima{s}'.format(s=arima_suffix),
                    rotation=90,
                    xticks=xticks
                )

                plot_ts(
                    data=test_predicted_df_long,
                    xlab='dates',
                    ylab='value',
                    hue='variable',
                    directory=directory,
                    filename='ts_test_arima{s}'.format(s=arima_suffix),
                    rotation=90,
                    xticks=xticks
                )

                # trend analysis
                decomposed = a[0].get_decomposed()
                decomposed.plot()
                plt.savefig(
                    '{d}/{f}'.format(
                        d=directory,
                        f='trend{suffix}'.format(suffix=arima_suffix)
                    )
                )

                if show:
                    plt.show()
                else:
                    plt.close()

    def lstm(
        self,
        normalize_key,
        directory='viz',
        plot=True,
        show=False,
        suffix=None,
        date_index='date',
        epochs=100,
        dropout=0.2,
        batch_size=32,
        validation_split=0,
        activation='linear',
        xticks=True
    ):

        '''

        implement arima model.

        '''

        # intialize
        l = model(
            df=self.df,
            model_type='lstm',
            normalize_key=normalize_key,
            date_index=date_index,
            epochs=epochs,
            dropout=dropout,
            batch_size=batch_size,
            validation_split=validation_split,
            activation=activation
        )

        # predict
        l.predict()
        self.model_scores['lstm'] = {
            'mse': l.get_mse(),
            'history': l.get_fit_history()
        }

        if plot:
            #
            # @train_actual, entire train values
            # @test_actual, entire train values
            # @predicted, only predicted values
            #
            l_data = l.get_data()
            l_predict_test = l.get_predict_test()
            train_actual = l_data['train_data']
            train_predicted = [x for x in l_predict_test[0]]
            test_actual = l_data['test_data']
            test_predicted = [x[0]
                if isinstance(x, (tuple, list, set, np.ndarray))
                else x for x in l_predict_test[1]]

            test_predicted_df = pd.DataFrame({
                'actual': test_actual[-len(test_predicted):],
                'predicted': test_predicted,
                'dates': l_data['test_index'][-len(test_predicted):]
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
                    'dates':  l_data['train_index']
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

    def get_model_scores(self, key=None):
        '''

        get current model scores for all implemented models.

        '''

        if key:
            return(self.model_scores[key])

        return(self.model_scores)
