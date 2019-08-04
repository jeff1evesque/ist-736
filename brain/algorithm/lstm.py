#!/usr/bin/python

import math
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class Lstm():
    '''

    apply lstm variant of recurrent neural network.

    '''

    def __init__(
        self,
        data,
        look_back=1,
        train=False
    ):
        '''

        define class variables.

        '''

        self.data = data
        self.look_back = look_back
        self.row_length = len(self.data)
        self.history = pd.Series()

        # sort dataframe by date
        self.data.sort_index(inplace=True)

        #
        # split and normalize
        #
        self.split_data()
        self.trainX, self.trainY = self.normalize(self.get_data()[0])
        self.testX, self.testY = self.normalize(self.get_data()[1])

        # train
        if train:
            self.train()

    def split_data(self, test_size=0.2):
        '''

        split data into train and test.

        Note: this requires execution of 'self.normalize'.

        '''

        # split without shuffling timeseries
        self.df_train, self.df_test = train_test_split(
            self.data,
            test_size=test_size,
            shuffle=False
        )
        self.history = self.df_train

    def get_data(self):
        '''

        get current train and test data.

        '''

        return(self.df_train, self.df_test)

    def normalize(self, data):
        '''

        normalization step: given vector [x], return [x, y] matrix:

            x       y
            112		118
            118		132
            132		129
            129		121
            121		135

        reshape step: train data to conform to lstm format requirements.
            convert current [samples, features] to required lstm format
            format [samples, timesteps, features].

        '''

        #
        # normalization step: utilize scaling normalization.
        #
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        dataset = self.scaler.fit_transform(data[:, np.newaxis])

        # eliminate edge cases
        if (self.look_back >= self.row_length):
            self.look_back = math.ceil(self.row_length / 4)

        # convert array of values into dataset matrix
        X, y = [], []
        for i in range(len(dataset) - self.look_back - 1):
            a = dataset[i:(i+self.look_back), 0]
            X.append(a)
            y.append(dataset[i + self.look_back, 0])

        #
        # reshape step
        #
        return(
            np.reshape(
                np.array(X),
                (np.array(X).shape[0], 1, np.array(X).shape[1])
            ),
            np.array(y)
        )

    def train(self, epochs=100, batch_size=32):
        '''

        train lstm model.

        '''

        # class variables
        self.epochs = epochs
        self.batch_size = batch_size

        # Initialize RNN
        self.regressor = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(
            units = 50,
            return_sequences = True,
            input_shape = (1, self.look_back)
        ))
        self.regressor.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(
            units = 50,
            return_sequences = True
        ))
        self.regressor.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(
            units = 50,
            return_sequences = True
        ))
        self.regressor.add(Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units = 50))
        self.regressor.add(Dropout(0.2))

        # Adding the output layer
        self.regressor.add(Dense(units = 1))

        # Compiling the RNN
        self.regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

        # Fitting the RNN to the Training set
        self.fit_history = self.regressor.fit(
            self.trainX,
            self.trainY,
            epochs = self.epochs,
            batch_size = self.batch_size
        )

    def update_train(self):
        '''

        update current train data with 'self.history'.

        '''

        self.df_train = self.history

    def get_lstm_params(self):
        '''

        return lstm parameters used during train.

        '''

        return(self.epochs, self.batch_size)

    def get_actual(self):
        '''

        return actual lagged values.

        '''

        return(
            self.scaler.inverse_transform([self.trainY]),
            self.scaler.inverse_transform([self.testY])
        )

    def predict(self, type='train'):
        '''

        generate prediction using hold out sample.

        @inverse_transform, convert prediction back to normal scale.

        '''

        if not hasattr(self, 'train_predict') or type == 'train':
            train_predict = self.regressor.predict(self.trainX)
            inverse_train_predict = self.scaler.inverse_transform(train_predict)

            self.train_predict = pd.Series(
                [x for x in self.df_train],
                index=self.df_train.index.values
            )

        test_predict = self.regressor.predict(self.testX)
        inverse_test_predict = self.scaler.inverse_transform(test_predict)

        #
        # rolling prediction: occurs when the overall history exceeds original
        #     data length.
        #
        if len(self.history) > len(self.data):
            history_idx = pd.date_range(
                self.history.tail(1).index[-1],
                periods=2,
                freq='D'
            )[1:]

            self.test_predict = pd.Series(
                [x[0] for x in inverse_test_predict],
                index=history_idx
            )

        else:
            self.test_predict = pd.Series(
                [x[0] for x in inverse_test_predict],
                index=self.df_test[:len(self.df_test) - self.look_back - 1].index.values
            )

        self.history = self.history.append(self.test_predict)

        return(self.train_predict, self.test_predict)

    def get_predict_test(self):
        '''

        return previous prediction result.

        '''

        return(self.train_predict, self.test_predict)

    def get_mse(self):
        '''

        return mean squared error.

        '''

        actual_train, actual_test = self.get_actual()

        try:
            train_score = math.sqrt(
                mean_squared_error(actual_train[0], self.train_predict[:,0])
            )
            test_score = math.sqrt(
                mean_squared_error(actual_test[0], self.test_predict[:,0])
            )
            return(train_score, test_score)

        except:
            return('No score available')

    def get_model(self):
        '''

        get trained lstm model.

        '''

        return(self.regressor)

    def get_fit_history(self, history_key=None):
        '''

        return model history object:

            {
                'acc': [0.9843952109499714],
                'loss': [0.050826362343496051],
                'val_acc': [0.98403786838658314],
                'val_loss': [0.0502210383056177]
            }

        '''

        if history_key:
            return(self.fit_history.history[history_key])
        return(self.fit_history.history)

    def get_index(self):
        '''

        get dataframe row index.

        '''

        return(self.data.index.values)
