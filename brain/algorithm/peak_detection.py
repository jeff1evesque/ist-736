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
        self.lag = lag
        self.influence = influence
        self.filteredY = np.array(self.data)
        self.this_file = os.path.basename(__file__)

        if isinstance(threshold, int):
            self.threshold = [threshold]
            self.signals = [[0] * len(self.data)]
            self.avg_filter = [[0] * len(self.data)]
            self.std_filter = [[0] * len(self.data)]
            self.avg_filter[0][self.lag - 1] = np.mean(self.data[0:self.lag])
            self.std_filter[0][self.lag - 1] = np.std(self.data[0:self.lag])

        elif isinstance(threshold, list):
            self.threshold = threshold
            self.signals = [[0] * len(self.data) for x in threshold]
            self.avg_filter = [[0] * len(self.data) for x in threshold]
            self.std_filter = [[0] * len(self.data) for x in threshold]

            for i in range(len(threshold)):
                self.avg_filter[i][self.lag - 1] = np.mean(self.data[0:self.lag])
                self.std_filter[i][self.lag - 1] = np.std(self.data[0:self.lag])

        else:
            print('Error ({f}): threshold must be int, or list of ints'.format(
                self.this_file
            ))
            exit(999)

    def add(self, new_value):
        '''

        add new value and update z-score model.

        '''

        self.data.append(new_value)
        for i,t in enumerate(self.threshold):
            self.filteredY += [0]
            self.signals[i] += [0]
            self.avg_filter[i] += [0]
            self.std_filter[i] += [0]

        self.update()

    def remove(self, indices):
        '''

        remove value at specified index and regenerate scores.

        '''

        l = range(len(self.filteredY))
        self.filteredY[:] = [self.filteredY[x] for x in l if x not in indices]
        self.update()

    def update(self):
        '''

        update series with new data, and return last value.

        '''

        idx = len(self.data) - 1

        if idx < self.lag:
            return(0)

        elif idx == self.lag:
            self.filteredY = np.array(self.data)
            self.signals = [[0] * len(self.data)]
            self.avg_filter = [[0] * len(self.data)]
            self.std_filter = [[0] * len(self.data)]
            self.avg_filter[0][self.lag - 1] = np.mean(self.data[0:self.lag])
            self.std_filter[0][self.lag - 1] = np.std(self.data[0:self.lag])
            return(0)

        for i,t in enumerate(self.threshold):
            current_val = abs(self.data[idx - 1] - self.avg_filter[i][idx - 1])
            threshold = t * self.std_filter[i][idx - 1]

            if current_val > threshold:
                if self.data[idx] > self.avg_filter[i][idx - 1]:
                    self.signals[i][idx] = 1
                else:
                    self.signals[i][idx] = -1

                self.filteredY[idx] = self.influence * self.data[idx] + \
                    (1 - self.influence) * self.filteredY[idx - 1]
                self.avg_filter[i][idx] = np.mean(self.filteredY[(idx - self.lag):idx])
                self.std_filter[i][idx] = np.std(self.filteredY[(idx - self.lag):idx])
            else:
                self.signals[i][idx] = 0
                self.filteredY[idx] = self.data[i]
                self.avg_filter[i][idx] = np.mean(self.filteredY[(idx - self.lag):idx])
                self.std_filter[i][idx] = np.std(self.filteredY[(idx - self.lag):idx])

        return(self.signals)

    def get_signals(self):
        '''

        get current signals.

        '''

        return([np.asarray(x) for x in self.signals])

    def get_avg_filter(self):
        '''

        get current average filter.

        '''

        return([np.asarray(x) for x in self.avg_filter])

    def get_std_filter(self):
        '''

        get current standard deviation filter.

        '''

        return([np.asarray(x) for x in self.std_filter])

    def get_data(self):
        '''

        get current data.

        '''

        return(self.data)
