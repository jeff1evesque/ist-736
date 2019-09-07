#!/usr/bin/python

import re
import pandas as pd
from pathlib import Path
from brain.algorithm.arima import Arima
from brain.algorithm.lstm import Lstm

def model(
    df,
    normalize_key,
    model_type='arima',
    date_index='date',
    epochs=100,
    dropout=0.2,
    batch_size=32,
    validation_split=0,
    activation='linear',
    num_cells=4,
    log_transform=0,
    auto_scale=False,
    rolling_grid_search=False,
    catch_grid_search=False
):
    '''

    return trained classifier.

    '''


    # sort dataframe by date
    df.sort_index(inplace=True)

    # initialize classifier
    if model_type == 'arima':
        model = Arima(
            data=pd.Series(
                df[normalize_key].values,
                [pd.Timestamp(x) for x in df[date_index].values]
            ),
            log_transform=log_transform
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
        model = Lstm(
            data=pd.Series(
                df[normalize_key].values,
                [pd.Timestamp(x) for x in df[date_index].values]
            )
        )
        model.train(
            epochs=epochs,
            dropout=dropout,
            batch_size=batch_size,
            validation_split=validation_split,
            activation=activation,
            num_cells=num_cells
        )

        return(model)
