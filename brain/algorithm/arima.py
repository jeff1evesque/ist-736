#!/usr/bin/python

import numpy as np
from math import log, ceil
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


class Arima():
    '''

    time series analysis using arima.

    '''

    def __init__(
        self,
        data,
        train=False,
        log_transform=0,
        iterations=1
    ):
        '''

        define class variables.

        '''

        self.data = data
        if log_transform:
            self.data = self.data.map(
                lambda a: log(a + log_transform)
            )

        # replace 'nan' with overall average
        self.data.dropna(inplace=True)
        self.row_length = len(self.data)

        # create train + test
        self.split_data()

        # train
        if train:
            self.train(iterations=iterations)

    def split_data(self, test_size=0.20):
        '''

        split data into train and test.

        Note: this requires execution of 'self.normalize'.

        '''

        # split without shuffling timeseries
        self.X_train, self.y_test = train_test_split(
            self.data,
            test_size=test_size,
            shuffle=False
        )
        self.df_train = self.X_train
        self.df_test = self.y_test

    def get_data(self):
        '''

        get current train and test data.

        '''

        return(self.df_train, self.df_test)

    def train(
        self,
        iterations=1,
        order=[1,0,0],
        rolling_grid_search=False,
        catch_grid_search=False
    ):
        '''

        train arima model.

        @order, (p,q,d) arguments can be defined using acf (MA), and pacf (AR)
            implementation. Corresponding large significant are indicators.

        @rolling_grid_search, implement default grid-search when 'True',
            otherwise implement 'auto_scale' grid search when 'auto'.

        @catch_grid_search, implement grid-search if exception raised during
            'model.fit'.

        Note: requires 'split_data' to be executed.

        '''

        actuals, predicted, rolling, differences = [], [], [], []
        self.history = self.df_train

        #
        # @order, if supplied through R, elements will be interpretted as float.
        #     For example c(1, 1, 0) will be interpretted as [1.0, 1.0, 0.0] in
        #     python, which breaks iterable indices.
        #
        self.order = [int(i) for i in order]
        for t in range(iterations):
            if (
                isinstance(rolling_grid_search, (list, set, tuple)) and
                len(rolling_grid_search) == 2
            ):
                self.order = self.grid_search(auto_scale=rolling_grid_search)[2]

            elif rolling_grid_search:
                self.order = self.grid_search()[2]


            model = ARIMA(self.history.tolist(), order=self.order)

            try:
                model_fit = model.fit(disp=0)
                fit_success = True

            except Exception as e:
                print('\n\n')
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('Warning: exception raised fitting model.')
                print('Message: {e}'.format(e=e))
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                fit_success = False

            try:
                if not fit_success and catch_grid_search:
                    self.order = self.grid_search()[2]
                    model = ARIMA(self.history, order=self.order)
                    model_fit = model.fit(disp=0)
                    fit_success = True

                elif not fit_success:
                    raise ValueError('No model fit, try setting catch_grid_search.')

            except Exception as e:
                print('\n\n')
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('Warning: cannot accomodate exception with grid-search.')
                print('Message: {e}'.format(e=e))
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                return(False)

            output = model_fit.forecast()
            yhat = float(output[0])
            predicted.append(yhat)

            #
            # observation: if current value doesn't exist from test, append current
            #     prediction, to ensure successive rolling prediction computed.
            #
            # @rolling, defined when 'iterations' exceeds original dataframe length,
            #     which defines the rolling predictions.
            #
            try:
                obs = float(self.df_test[t])
                actuals.append(obs)
                differences.append(abs(1-(yhat/obs)))

            except:
                obs = yhat
                rolling.append(obs)

            # rolling prediction data
            self.history.append(pd.Series([obs]), ignore_index=True)

        self.differences = (actuals, predicted, differences)
        self.mse = mean_squared_error(actuals, predicted)
        self.rolling = rolling

        return(True)

    def grid_search(
        self,
        p_values=range(0,3),
        d_values=range(0,3),
        q_values=range(0,3),
        auto_scale=None
    ):
        '''

        determine optimal arima arguments using (p,q,d) range.

        @auto_scale, dynamically generate (p,q,d) based on data length.
            [0], minimum threshold required for auto-scaling
            [1], scaling factor, determines the (p,q,d) range

        '''

        try:
            data = self.history

        except:
            data = self.data

        if (
            auto_scale and
            len(auto_scale) == 2 and
            all(isinstance(x, (int, float)) for x in auto_scale) and
            len(data) > auto_scale[0]
        ):
            auto_range = ceil(len(data) * auto_scale[1])
            p_value=range(0, auto_range)
            d_value=range(0, auto_range)
            q_value=range(0, auto_range)

        best_adf, best_score, best_pqd = float('inf'), float('inf'), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    try:
                        model = self.train(order=order)
                        mse = self.get_mse()
                        adf = self.get_adf()

                        if (
                            mse < best_score and
                            np.isfinite(adf[0]) and
                            np.isfinite(adf[1])
                         ):
                            best_adf = adf
                            best_score = mse
                            best_pqd = order

                    except Exception as e:
                        print('\n\n')
                        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                        print('Warning: exception raised running grid-search.')
                        print('Message: {e}'.format(e=e))
                        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                        continue

        return(best_adf, best_score, best_pqd)

    def get_order(self):
        '''

        return arima (p,d,q) parameters.

        '''

        return(self.order)

    def get_mse(self):
        '''

        return mean squared error on trained arima model.

        '''

        return(self.mse)

    def get_difference(self, data=None, diff=0):
        '''

        return differenced timeseries.

        '''

        # default to arima (p,d,q) parameters
        if self.order:
            d  = self.order[1]
        else:
            d = diff

        # fallback on train/test split
        if data:
            original = data
        else:
            original = self.df_test

        # determine difference
        if int(diff) > 0:
            interval = int(d)
            differenced = []

            for i in range(interval, len(original)):
                value = original[i] - original[i - interval]
                differenced.append(value)

            return(differenced)

        return(original)

    def get_adf(self, data=None):
        '''

        return augmented dickey-fuller test:

            p-value > 0.05, fail to reject the null hypothesis, data has a unit
                unit root and is non-stationary.

            p-value <= 0.05, reject the null hypothesis, data does not have a
                unit root and is stationary.

        '''

        if not data:
            # ensure adf matches arima integrated difference
            if self.order:
                data = self.get_difference()

            else:
                data = self.df_test

        elif not data:
            data = 'Provide valid list'

        try:
            result = adfuller(data)

        except Exception as e:
            result = -999
            print(e)

        return(result)

    def get_differences(self):
        '''

        return differences between prediction against corresponding actual.

        '''

        return(self.differences)

    def get_rolling(self):
        '''

        return rolling predictions.

        '''

        return(self.rolling)

    def get_history(self):
        '''

        return entire timeseries history including rolling predictions.

        '''

        return(self.history)

    def get_decomposed(self, series=None, model='additive', freq=1):
        '''

        decompose a time series into original, trend, seasonal, residual.

        '''

        if not series:
            series = self.data

        result = seasonal_decompose(series, model=model, freq=freq)

        return(result)
