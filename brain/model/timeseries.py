#!/usr/bin/python

import re
import pandas as pd
from pathlib import Path
from brain.algorithm.arima import Arima
from brain.algorithm.lstm import Lstm
from config import model_config as c


def model(
    df,
    normalize_key,
    model_type='arima',
    date_index='date',
    log_delta=0.01,
    auto_scale=None,
    rolling_grid_search=False,
    catch_grid_search=False
):
    '''

    return trained classifier.

    @date_index, when provided the model will use this as the timeseries
        index. If not provided, the current dataframe index will be used.

    '''


    # sort dataframe by date
    df.sort_index(inplace=True)

    # initialize model
    if model_type == 'arima':
        if date_index and date_index in df:
            model = Arima(
                data=pd.Series(
                    df[normalize_key].values,
                    [pd.Timestamp(x) for x in df[date_index].tolist()]
                ),
                log_delta=log_delta
            )

        else:
            model = Arima(
                data=pd.Series(
                    df[normalize_key].values,
                    [pd.Timestamp(x) for x in df.index.values]
                ),
                log_delta=log_delta
            )

        # induce stationarity
        result = model.grid_search(auto_scale=auto_scale)

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
        if (
            isinstance(result[0], (list, set, tuple)) and
            len(result[0]) >= 2 and
            result[0][1] <= 0.05
        ):

            iterations = len(model.get_data('test'))
            success = model.train(
                iterations=iterations,
                order=result[2],
                rolling_grid_search=rolling_grid_search,
                catch_grid_search=catch_grid_search
            )

            if not success:
                return(False)

            if rolling_grid_search:
                return(model, self.get_order())
            return(model, result[2])

        else:
            return(False)

    elif model_type == 'lstm':
        if date_index and date_index in df:
            model = Lstm(
                data=pd.Series(
                    df[normalize_key].values,
                    [pd.Timestamp(x) for x in df[date_index].tolist()]
                ),
                n_steps_in=c['lstm_steps_in'],
                n_steps_out=c['lstm_steps_out']
            )

        else:
            model = Lstm(
                data=pd.Series(
                    df[normalize_key].values,
                    [pd.Timestamp(x) for x in df.index.values]
                ),
                n_steps_in=c['lstm_steps_in'],
                n_steps_out=c['lstm_steps_out']
            )

        if model.get_status(type='train_flag'):
            model.train(
                epochs=c['lstm_epochs'],
                dropout=c['lstm_dropout'],
                batch_size=c['lstm_batch_size'],
                validation_split=c['lstm_validation_split'],
                activation=c['lstm_activation'],
                num_cells=c['lstm_num_cells'],
                units=c['lstm_units']
            )

        return(model)
