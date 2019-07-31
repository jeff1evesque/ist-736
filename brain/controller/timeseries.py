#!/usr/bin/python

import math
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
        auto_scale=False,
        rolling_grid_search=False,
        catch_grid_search=False
    ):

        # local variables
        self.df = df
        self.model_scores = {}

        if suffix:
            self.suffix = '_{suffix}'.format(suffix=suffix)
        else:
            self.suffix=''

        # implement models
        if flag_arima:
            self.arima(
                df,
                normalize_key=normalize_key,
                log_transform=0.01,
                date_index=date_index,
                auto_scale=auto_scale,
                rolling_grid_search=rolling_grid_search,
                catch_grid_search=catch_grid_search,
                directory=directory
            )

        if flag_lstm:
            self.lstm(
                df,
                normalize_key=normalize_key,
                date_index=date_index,
                epochs=lstm_epochs,
                directory=directory
            )

    def arima(
        self,
        df,
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
            data=df,
            normalize_key=normalize_key,
            log_transform=log_transform,
            model_type='arima',
            date_index=date_index,
            auto_scale=auto_scale,
            rolling_grid_search=rolling_grid_search,
            catch_grid_search=catch_grid_search
        )

        if a and len(a) == 2:
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
                        data=a[0].get_data(
                            key=normalize_key,
                            key_to_list='True'
                        )[0],
                        diff=diff
                    )

                else:
                    train_actual = a[0].get_data(
                        key=normalize_key,
                        key_to_list='True'
                    )[0]

                test_actual = a[0].get_differences()[0]
                predicted = a[0].get_differences()[1]

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
        df,
        normalize_key,
        directory='viz',
        plot=True,
        show=False,
        suffix=None,
        date_index='date',
        epochs=100,
        xticks=True
    ):

        '''

        implement arima model.

        '''

        # intialize
        l = model(
            data=df,
            normalize_key=normalize_key,
            model_type='lstm',
            date_index=date_index,
            epochs=epochs
        )

        # predict
        l.predict()
        self.model_scores['lstm'] = {
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

    def get_model_scores(self, key=None):
        '''

        get current model scores for all implemented models.

        '''

        if key:
            return(self.model_scores[key])

        return(self.model_scores)
