#!/usr/bin/python

import math
import numpy as np
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

    def __init__(self, data, look_back=1, train=False, normalize_key=None):
        '''

        define class variables.

        '''

        self.look_back = look_back

        if isinstance(data, dict):
            self.data = pd.DataFrame(data)
        else:
            self.data = data

        self.row_length = len(self.data)

        # sort dataframe by date
        self.data['date'] = pd.to_datetime(self.data.date)
        self.data.sort_values(by=['date'], inplace=True)

        # convert column to dataframe index
        self.data.set_index('date', inplace=True)

        # convert dataframe columns to integer
        self.data.total = self.data.total.astype(int)

        # create train + test
        self.split_data()

        if normalize_key:
            self.normalize_key = normalize_key
            train_x, self.trainY = self.normalize(self.df_train)
            test_x, self.testY = self.normalize(self.df_test)

            #
            # reshape for lstm: convert current [samples, features] to required lstm 
            #     format [samples, timesteps, features].
            #
            self.trainX = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
            self.testX = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

        else:
            self.normalize_key = None

        # train
        if train:
            self.train()
            self.predict_test()

    def split_data(self, test_size=0.2):
        '''

        split data into train and test.

        Note: this requires execution of 'self.normalize'.

        '''

        # split without shuffling timeseries
        self.train, self.test = train_test_split(self.data, test_size=test_size, shuffle=False)
        self.df_train = pd.DataFrame(self.train)
        self.df_test = pd.DataFrame(self.test)

    def get_data(self, key=None, key_to_list=False):
        '''

        get current train and test data.

        '''

        if key:
            if key_to_list:
                return(self.df_train[key].tolist(), self.df_test[key].tolist())
            return(self.df_train[key], self.df_test[key])
        return(self.df_train, self.df_test)

    def normalize(self, data):
        '''

        given a vector [x], a matrix [x, y] is returned:

            x     y
            112		118
            118		132
            132		129
            129		121
            121		135

        @train_set, must be the value column from the original dataframe.

        '''

        # scaling normalization
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        dataset = self.scaler.fit_transform(data[[self.normalize_key]])

        # eliminate edge cases
        if (self.look_back >= self.row_length):
            self.look_back = math.ceil(self.row_length / 4)

        # convert array of values into dataset matrix
        X_train, y_train = [], []
        for i in range(len(dataset) - self.look_back - 1):
            a = dataset[i:(i+self.look_back), 0]
            X_train.append(a)
            y_train.append(dataset[i + self.look_back, 0])

        return(np.array(X_train), np.array(y_train))

    def train_model(self, epochs=100, batch_size=32):
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

    def predict_test(self, timesteps=10):
        '''

        generate prediction using hold out sample.

        '''

        train_predict = self.regressor.predict(self.trainX)
        test_predict = self.regressor.predict(self.testX)

        #
        # @inverse_transform, convert prediction back to normal scale.
        #
        self.train_predict = self.scaler.inverse_transform(train_predict)
        self.test_predict = self.scaler.inverse_transform(test_predict)

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
            train_score = math.sqrt(mean_squared_error(actual_train[0], self.train_predict[:,0]))
            test_score = math.sqrt(mean_squared_error(actual_test[0], self.test_predict[:,0]))
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
