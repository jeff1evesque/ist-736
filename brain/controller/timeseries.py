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
        directory_arima='viz/arima',
        directory_lstm='viz/lstm',
        directory_lstm_model='model/lstm',
        flag_arima=True,
        flag_lstm=True,
        plot=True,
        show=False,
        suffix=None,
        date_index='date',
        diff=1,
        xticks=True,
        arima_auto_scale=None,
        lstm_save=False,
        lstm_save_log=False,
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

        #
        # implement models
        #
        if flag_arima:
            self.arima(
                normalize_key=normalize_key,
                log_delta=0.01,
                date_index=date_index,
                auto_scale=arima_auto_scale,
                rolling_grid_search=rolling_grid_search,
                catch_grid_search=catch_grid_search,
                directory=directory_arima,
                suffix=suffix
            )

        if flag_lstm:
            self.lstm(
                normalize_key=normalize_key,
                date_index=date_index,
                directory=directory_lstm,
                directory_lstm_model=directory_lstm_model,
                suffix=suffix,
                save_model=lstm_save,
                save_model_log=lstm_save_log
            )

    def arima(
        self,
        normalize_key,
        directory='viz/arima',
        plot=True,
        show=False,
        suffix=None,
        date_index='date',
        diff=1,
        xticks=True,
        auto_scale=None,
        log_delta=0.01,
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
            log_delta=log_delta,
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
                # @train_actual, entire train values
                # @test_actual, entire train values
                # @predicted, only predicted values
                #
                if diff > 1:
                    train_actual = a[0].get_difference(
                        data=a[0].get_data('train'),
                        diff=diff
                    )

                else:
                    train_actual = a[0].get_data('train')

                test_actual = a[0].get_differences('test')
                predicted = a[0].get_differences('predicted')

                test_predicted_df = pd.DataFrame({
                    'actual': test_actual,
                    'predicted': predicted[:len(test_actual)],
                    'dates': a[0].get_data('test_index')
                })
                test_predicted_df_long = pd.melt(
                    test_predicted_df,
                    id_vars=['dates'],
                    value_vars=['actual', 'predicted']
                )

                #
                # overall plot
                #
                plot_ts(
                    data=pd.DataFrame({
                        'values': train_actual,
                        'dates': a[0].get_data('train_index')
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

                #
                # trend analysis
                #
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
        directory='viz/lstm',
        directory_lstm_model='model/lstm',
        plot=True,
        show=False,
        suffix=None,
        date_index='date',
        xticks=True,
        save_model=False,
        save_model_log=True
    ):

        '''

        implement arima model.

        '''

        # intialize
        l = model(
            df=self.df,
            model_type='lstm',
            normalize_key=normalize_key,
            date_index=date_index
        )

        #
        # predict
        #
        if l.get_status(type='train_flag'):
            l.predict()
            self.model_scores['lstm'] = {
                'mse': l.get_mse('test'),
                'history': l.get_fit_history()
            }

        if plot and l.get_status(type='train_flag'):
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

            #
            # overall plot
            #
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

        #
        # save model
        #
        if save_model and isinstance(save_model, bool):
            l.save(file_path='{a}/lstm{b}.h5'.format(
                a=directory_lstm_model,
                b=suffix
            ))

        #
        # save model log
        #
        if save_model_log and isinstance(save_model_log, bool):
            loss_df = pd.DataFrame({
                'train': l.get_fit_history('loss'),
                'validation': l.get_fit_history('val_loss'),
                'epoch': range(1, len(l.get_fit_history('loss')) + 1)
            })

            loss_df_long = pd.melt(
                loss_df,
                id_vars=['epoch'],
                value_vars=['loss', 'val_loss']
            )

            plot_ts(
                data=loss_df_long,
                xlab='epoch',
                ylab='loss',
                hue='variable',
                directory=directory_lstm_model,
                filename=suffix,
                rotation=90,
                xticks=xticks
            )

            l.get_fit_history().to_csv('{a}/lstm{b}.csv'.format(
                a=directory_lstm_model,
                b=suffix
            ))

        #
        # reset memory
        #
        l.reset_memory()

    def get_model_scores(self, key=None):
        '''

        get current model scores for all implemented models.

        '''

        if key:
            return(self.model_scores[key])

        return(self.model_scores)
