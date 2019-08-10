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
        trainX, self.trainY = self.normalize(self.get_data()[0])
        testX, self.testY = self.normalize(self.get_data()[1])

        print('11111111111111111111111111111111111111111111111111')
        print(self.trainY)
        print('11111111111111111111111111111111111111111111111111')

        self.trainX = self.reshape(trainX),
        self.testX = self.reshape(testX)

        print('11111111111111111111111111111111111111111111111111')
        print(self.trainX)
        print('11111111111111111111111111111111111111111111111111')

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

    def get_data(self, remove_lookup=False):
        '''

        get current train and test data.

        @remove_lookup, model train lagged original timeseries by a 'look_back'
            factor. To account for this, the original test split is trimmed by
            this same factor.

        '''

        if remove_lookup:
            return(
                self.df_train,
                self.df_test[:len(self.df_test) - self.look_back - 1]
            )

        return(self.df_train, self.df_test)

    def split_sequence(self, sequence, n, m):
        '''

        use last n steps as input to forecast next m steps.

            sequence = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

            let n = 4, m = 2

        Therefore,
            [5, 10, 15, 20] [25, 30]
            [25, 30, 35, 40] [45, 50]

        '''

        X, y = [], []
        for i in range(len(sequence)):
            # subsample indices
	        n_end_index = i + n
            m_end_index = n_end_index + m

            # cannot exceed sequence
            if m_end_index > len(sequence):
                break

            # aggregate subsamples
            seq_x = sequence[i:n_end_index]
            seq_y = sequence[n_end_index:m_end_index]
            X.append(seq_x)
            y.append(seq_y)

        return(np.array(X), np.array(y))

    def reshape(self, x):
        '''

        reshape [samples, features] to [samples, timesteps, features], which is
        required by the implemented LSTM model.

        '''

        return(np.reshape(x.shape[0], 1, x.shape[1]))

    def train(self, epochs=100, batch_size=32):
        '''

        train lstm model.

        @LSTM, units - the number of neurons to apply for the given layer
        @LSTM, return sequences - when False, returns (batch_size, units), 3D
        @LSTM, input_shape - tuple (time_steps, num_input_units), which is an
            array, (batch_size, time_steps, num_input_units), where any batch
            size can be used. Likewise, the 'input_shape' can be replaced with
            the 'batch_input_shape', which implements a non-arbitrary batch
            size parameter, (8, time_steps, num_input_units).

        @Dropout, each layer ignores 20% of neurons to reduce overfitting.

        '''

        # class variables
        self.epochs = epochs
        self.batch_size = batch_size

        # initialize rnn
        self.regressor = Sequential()

        #
        # first LSTM layer with Dropout regularisation
        #
        self.regressor.add(LSTM(
            units = 50,
            return_sequences = True,
            input_shape = (1, self.look_back)
        ))
        self.regressor.add(Dropout(0.2))

        # second LSTM layer with Dropout regularisation
        self.regressor.add(LSTM(
            units = 50,
            return_sequences = True
        ))
        self.regressor.add(Dropout(0.2))

        # third LSTM layer with Dropout regularisation
        self.regressor.add(LSTM(
            units = 50,
            return_sequences = True
        ))
        self.regressor.add(Dropout(0.2))

        # fourth LSTM layer with Dropout regularisation
        self.regressor.add(LSTM(
            units = 50,
            return_sequences = True
        ))
        self.regressor.add(Dropout(0.2))

        #
        # output layer: only one neuron, since only one value predicted.
        #
        self.regressor.add(Dense(units = 1))

        # compile the RNN
        self.regressor.compile(
            optimizer = 'adam',
            loss = 'mean_squared_error'
        )

        # fit RNN to train data
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
