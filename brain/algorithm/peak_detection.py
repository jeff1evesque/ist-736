#!/usr/bin/python

import numpy as np


class PeakDetection():
    '''

    Using provided timeseries, determine which points exceed defined threshold.

    Note: code was adapted from https://stackoverflow.com/a/56451135.

    '''

    def __init__(self, data, lag=3, threshold=1, influence=0.5):
        '''

        instantiate series and threshold.

        @data, univariate list.
        @lag, moving windows.
        @threshold, z-score (standard deviation) threshold, when exceeded
            the corresponding value is considered signal.
        @influence, weighting factor [0,1] for signal values, relative to
            original datapoints. If 'influence=0.5', then signal values have
            0.5 influence relative to original datapoints when recaculating
            the new threshold.

        '''

        self.y = list(data)
        self.length = len(self.y)
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.signals = [0] * len(self.y)
        self.filteredY = np.array(self.y).tolist()
        self.avgFilter = [0] * len(self.y)
        self.stdFilter = [0] * len(self.y)
        self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
        self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()

    def update(self, new_value):
        '''

        update series with new data, and return last value.

        '''

        self.y.append(new_value)
        i = len(self.y) - 1
        self.length = len(self.y)

        if i < self.lag:
            return(0)
        elif i == self.lag:
            self.signals = [0] * len(self.y)
            self.filteredY = np.array(self.y).tolist()
            self.avgFilter = [0] * len(self.y)
            self.stdFilter = [0] * len(self.y)
            self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
            self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()
            return(0)

        self.signals += [0]
        self.filteredY += [0]
        self.avgFilter += [0]
        self.stdFilter += [0]

        current_val = abs(self.y[i] - self.avgFilter[i - 1])
        threshold = self.threshold * self.stdFilter[i - 1]
        if current_val > threshold:
            if self.y[i] > self.avgFilter[i - 1]:
                self.signals[i] = 1
            else:
                self.signals[i] = -1

            self.filteredY[i] = self.influence * self.y[i] + \
                (1 - self.influence) * self.filteredY[i - 1]
            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])
        else:
            self.signals[i] = 0
            self.filteredY[i] = self.y[i]
            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])

        return(self.signals[i])

    def get_signals(self):
        '''

        get current signals.

        '''

        return(self.signals)

    def get_avgfilter(self):
        '''

        get current average filter.

        '''

        return(self.avgFilter)

    def get_avgfilter(self):
        '''

        get current standard deviation filter.

        '''

        return(self.stdFilter)
