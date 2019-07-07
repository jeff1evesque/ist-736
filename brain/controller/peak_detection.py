#!/usr/bin/python

import numpy as np
from brain.algorithm.peak_detection import PeakDetection
import matplotlib.pyplot as plt


def peak_detection(
    data,
    directory='viz',
    suffix='',
    plot=True,
    show=False
):
    '''

    implement granger test for causality.

    '''

    if suffix:
        suffix = '_{a}'.format(a=suffix)

    peaks = PeakDetection()
    signals = peaks.get_signals()
    stdFilter = peaks.get_stdfilter()
    avgFilter = peaks.get_avgFilter()

    if plot:
        plt.subplot(211)
        plt.plot(np.arange(1, len(y)+1), y)

        plt.plot(
            np.arange(1, len(y)+1),
            avgFilter
            color='cyan',
            lw=2
        )

        plt.plot(
            np.arange(1, len(y)+1),
            avgFilter + threshold * stdFilter,
            color='green',
            lw=2
        )

        plt.plot(
            np.arange(1, len(y)+1),
            avgFilter - threshold * stdFilter,
            color='green',
            lw=2
        )

        plt.subplot(212)
        plt.step(
            np.arange(1, len(y)+1),
            signals,
            color='red',
            lw=2
        )
        plt.ylim(-1.5, 1.5)

        plt.savefig('{d}/peak_detection{suffix}'.format(suffix=suffix))

        if show:
            plt.show()
        else:
            plt.close()

    return(signals)
