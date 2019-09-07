#!/usr/bin/python

import math
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class Lstm():
    '''

    apply lstm variant of recurrent neural network.

    Note: keras==2.1.2 is required.

    '''

    def __init__(
        self,
        data,
        n_steps_in=4,
        n_steps_out=1,
        train=False,
        type='univariate'
    ):
        '''

        define class variables.

        '''

        self.type = type

        #
        # cleanse data: sort univariate, and replace 'nan' with average.
        #
        if self.type != 'multivariate':
            self.data = data
            self.data.sort_index(inplace=True)
            self.data.fillna(self.data.mean(), inplace=True)

        #
        # keep track of data
        #
        self.history = self.data
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out

        self.data = [x[0] if isinstance(x, (list, set, tuple, np.ndarray))
            else x
            for x in data.values.reshape(
                len(data.values),
                1
            )
        ]

        #
        # split sequence
        #
        self.split_data(scale=True)

        #
        # conditionally define uni/multi-variate
        #
        if self.type == 'multivariate':
            self.n_features = X1.shape[2]

            #
            # TODO: incomplete
            #

        else:
            # univariate: one feature
            self.n_features = 1

            #
            # reshape
            #
            X1 = np.array([[[a] for a in x] for x in self.trainX])
            X2 = np.array([[[a] for a in x] for x in self.testX])

            self.trainX = X1.reshape(
                X1.shape[0],
                X1.shape[1],
                self.n_features
            )

            self.testX = X2.reshape(
                X2.shape[0],
                X2.shape[1],
                self.n_features
            )

        # train
        if train:
            self.train()

    def scale(self, data=None, feature_range=(-1, 1)):
        '''

        Current train and test data is scaled, to better help convergence.
        Without a scaled dataset, convergence may take a long time, especially
        if variance is high. When data has high order of variance, convergence
        may not even occur.

        @feature_range, range of transformed scale, with zero mean.

        Note: this requires split_data to have run.

        '''

        if data is None:
            data = self.data

        #
        # fit scaler: return fitted scaler with data transformed.
        #
        # Note: scaler requires 2D array
        #
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled = scaler.fit_transform(data)

        return(scaler, scaled)

    def invert_scale(self, scaler, X):
        '''

        inverse scale predicted value

        '''

        return(scaler.inverse_transform(X))

    def split_data(self, data=None, test_size=0.2, scale=False):
        '''

        split data into train and test.

        Note: this requires execution of 'self.normalize'.

        '''

        # split without shuffling timeseries
        if data:
            train, test = train_test_split(
                data,
                test_size=test_size,
                shuffle=False
            )

            if scale:
                X1, self.trainY = self.split_sequence(
                    train,
                    n=self.n_steps_in,
                    m=self.n_steps_out
                )
                scalerX = self.scale(X1)
                scalerY = self.scale(self.trainY)
                self.scalerX = scalerX[0]
                self.scalerY = scalerY[0]

                self.trainX = scalerX[1]
                self.trainY = scalerY[1]

                X2, self.testY = self.split_sequence(
                    test,
                    n=self.n_steps_in,
                    m=self.n_steps_out
                )

                self.testX = self.scalerX.transform(X2)
                self.testY = self.scalerY.transform(self.testY)

        else:
            self.df_train, self.df_test = train_test_split(
                self.data,
                test_size=test_size,
                shuffle=False
            )

            if scale:
                X1, self.trainY = self.split_sequence(
                    self.df_train,
                    n=self.n_steps_in,
                    m=self.n_steps_out
                )
                scalerX = self.scale(X1)
                scalerY = self.scale(self.trainY)
                self.scalerX = scalerX[0]
                self.scalerY = scalerY[0]

                self.trainX = scalerX[1]
                self.trainY = scalerY[1]

                X2, self.testY = self.split_sequence(
                    self.df_test,
                    n=self.n_steps_in,
                    m=self.n_steps_out
                )

                self.testX = self.scalerX.transform(X2)
                self.testY = self.scalerY.transform(self.testY)

            else:
                self.trainX, self.trainY = self.split_sequence(
                    train,
                    n=self.n_steps_in,
                    m=self.n_steps_out
                )

                self.testX, self.testY = self.split_sequence(
                    test,
                    n=self.n_steps_in,
                    m=self.n_steps_out
                )

    def get_data(self, type=None):
        '''

        get current train and test data.

        '''

        if type == 'train_index':
            return(self.history[:len(self.df_train)].index)

        elif type == 'test_index':
            return(self.history[-len(self.df_test):].index)

        elif type == 'train_data':
            return(self.df_train)

        elif type == 'test_data':
            return(self.df_test)

        else:
            return({
                'train_index': self.history[:len(self.df_train)].index,
                'test_index': self.history[-len(self.df_test):].index,
                'train_data': self.df_train,
                'test_data': self.df_test
            })

    def get_predict_test(self):
        '''

        return previous prediction result.

        Note: original data was scaled using minmaxscaler.

        '''

        return(
            self.invert_scale(self.scalerX, self.train_predict),
            self.invert_scale(self.scalerY, self.test_predict)
        )

    def get_actual(self):
        '''

        get lagged values.

        Note: original data was scaled using minmaxscaler.

        '''

        return(
            [self.invert_scale(self.scalerY, self.trainY)],
            [self.invert_scale(self.scalerY, self.testY)]
        )

    def split_sequence(self, sequence, n, m=1):
        '''

        split univariate sequence into samples, use last n steps as input to
        forecast next m steps.

            sequence = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

            let n = 4, m = 2

        Therefore,

            [5, 10, 15, 20] [25, 30]
            [10, 15, 20, 25] [30, 35]
            [15, 20, 25, 30] [35, 40]
            [20, 25, 30, 35] [40, 45]
            [25, 30, 35, 40] [45, 50]

        '''

        X, y = [], []
        for i in range(len(sequence)):
            # subsample indices
            n_end_index = i + n
            m_end_index = n_end_index + m

            # cannot exceed sequence
            if m_end_index > len(sequence) - 1:
                break

            # aggregate subsamples
            if self.type == 'multivariate':
                seq_x = sequence[i:n_end_index]
                seq_y = sequence[n_end_index]

                #
                # TODO: incomplete
                #

            else:
                seq_x = sequence[i:n_end_index]
                seq_y = sequence[n_end_index:m_end_index]

            X.append(seq_x)
            y.append(seq_y)

        return(np.array(X), np.array(y))

    def reshape(self, x, n_features):
        '''

        reshape [samples, features] to [samples, timesteps, features].

        @samples, the number of data, or how many rows in your data set
        @timesteps, the number of times to feed in the model or LSTM
        @features,  the number of columns of each sample

        '''

        return(x.reshape((x.shape[0], x.shape[1], n_features)))

    def train(
        self,
        epochs=100,
        dropout=0.2,
        batch_size=32,
        validation_split=0.2,
        activation='sigmoid',
        num_cells=4
    ):
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
        @activation, 'linear' is generally preferred for linear regression,
            'sigmoid' for nonlinear regression, and 'softmax' is better for
            classification.

        '''

        # class variables
        self.epochs = epochs
        self.batch_size = batch_size

        # initialize rnn
        self.regressor = Sequential()

        #
        # lstm cell with dropout regularization
        #
        for cell in range(num_cells):
            self.regressor.add(LSTM(
                units = 50,
                return_sequences = True,
                input_shape = (
                    self.n_steps_in,
                    self.n_features
                )
            ))
            self.regressor.add(Dropout(dropout))

        #
        # output layer: only one neuron, since only one value predicted.
        #
        self.regressor.add(Dense(
            units = 1,
            activation = activation
        ))

        # compile the RNN
        self.regressor.compile(
            optimizer = 'adam',
            loss = 'mse'
        )

        # fit RNN to train data
        self.fit_history = self.regressor.fit(
            self.trainX,
            self.trainY,
            epochs = self.epochs,
            batch_size = self.batch_size,
            verbose = 1,
            validation_split = validation_split
        )

    def get_lstm_params(self):
        '''

        return lstm parameters used during train.

        '''

        return(self.epochs, self.batch_size)

    def predict(self, type='train', verbose=0):
        '''

        generate prediction using hold out sample.

        Note: since train X is scaled, and target y is not, prediction is not
              invert-transformed.

        '''

        self.train_predict = []
        for data in self.trainX:
            train_input = np.array([x[0] for x in data])
            train_reshaped = train_input.reshape((
                1,
                self.n_steps_in,
                self.n_features
            ))
            predicted = self.regressor.predict(train_reshaped, verbose=verbose)
            self.train_predict.append(predicted[0][0])

        self.test_predict = []
        for data in self.testX:
            test_input = np.array([x[0] for x in data])
            test_reshaped = test_input.reshape((
                1,
                self.n_steps_in,
                self.n_features
            ))
            predicted = self.regressor.predict(test_reshaped, verbose=verbose)
            self.test_predict.append(predicted[0][0])

        return(self.train_predict, self.test_predict)

    def get_mse(self):
        '''

        return mean squared error.

        '''

        actual_train, actual_test = self.get_actual()

        try:
            train_score = math.sqrt(
                mean_squared_error(actual_train[0], self.train_predict)
            )
            test_score = math.sqrt(
                mean_squared_error(actual_test[0], self.test_predict)
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
