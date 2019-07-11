#!/usr/bin/python

import os.path
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

        self.data = list(data)
        self.length = len(self.data)
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.filteredY = np.array(self.data).tolist()
        self.this_file = os.path.basename(__file__)

        if isinstance(threshold, int):
            self.signals = [[0] * len(self.data)]
            self.avgFilter = [[0] * len(self.data)]
            self.stdFilter = [[0] * len(self.data)]

            self.avgFilter[0][self.lag - 1] = np.mean(self.data[0:self.lag]).tolist()
            self.stdFilter[0][self.lag - 1] = np.std(self.data[0:self.lag]).tolist()

        elif isinstance(threshold, list):
            self.signals = [[0] * len(self.data) for x in threshold]
            self.avgFilter = [[0] * len(self.data) for x in threshold]
            self.stdFilter = [[0] * len(self.data) for x in threshold]

            for i in range(len(threshold)):
                self.avgFilter[i][self.lag - 1] = np.mean(self.data[0:self.lag]).tolist()
                self.stdFilter[i][self.lag - 1] = np.std(self.data[0:self.lag]).tolist()

        else:
            print('Error ({f}): threshold must be int, or list of ints'.format(
                self.this_file
            ))
            exit(999)

    def update(self, new_value):
        '''

        update series with new data, and return last value.

        '''

        self.data.append(new_value)
        idx = len(self.data) - 1
        self.length = len(self.data)

        if idx < self.lag:
            return(0)
        elif idx == self.lag:
            self.filteredY = np.array(self.data).tolist()
            self.signals = [[0] * len(self.data)]
            self.avgFilter = [[0] * len(self.data)]
            self.stdFilter = [[0] * len(self.data)]
            self.avgFilter[0][self.lag - 1] = np.mean(self.data[0:self.lag]).tolist()
            self.stdFilter[0][self.lag - 1] = np.std(self.data[0:self.lag]).tolist()
            return(0)

        for i in range(len(self.threshold)):
            self.signals[i] += [0]
            self.filteredY[i] += [0]
            self.avgFilter[i] += [0]
            self.stdFilter[i] += [0]

            current_val = abs(self.data[idx] - self.avgFilter[i][idx - 1])
            threshold = self.threshold * self.stdFilter[i][idx - 1]

            if current_val > threshold:
                if self.data[idx] > self.avgFilter[i][idx - 1]:
                    self.signals[i][idx] = 1
                else:
                    self.signals[i][idx] = -1

                self.filteredY[idx] = self.influence * self.data[idx] + \
                    (1 - self.influence) * self.filteredY[idx - 1]
                self.avgFilter[i][idx] = np.mean(self.filteredY[(idx - self.lag):idx])
                self.stdFilter[i][idx] = np.std(self.filteredY[(idx - self.lag):idx])
            else:
                self.signals[i][idx] = 0
                self.filteredY[i] = self.data[i]
                self.avgFilter[i][idx] = np.mean(self.filteredY[(i - self.lag):i])
                self.stdFilter[i][idx] = np.std(self.filteredY[(i - self.lag):i])

        return(self.signals)

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

    def get_stdfilter(self):
        '''

        get current standard deviation filter.

        '''

        return(self.stdFilter)

    def get_data(self):
        '''

        get current data.

        '''

        return(self.data)
