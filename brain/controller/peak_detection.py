#!/usr/bin/python

import math
import numpy as np
from brain.algorithm.peak_detection import PeakDetection
import matplotlib.pyplot as plt


def peak_detection(
    data,
    directory='viz',
    suffix='',
    plot=True,
    show=False,
    threshold=None,
    auto=True
):
    '''

    implement z-score peak detection, with default behavior if possible.

    '''

    if suffix:
        suffix = '_{a}'.format(a=suffix)

    #
    # threshold: autogenerate if not provided, cancel if not enough data.
    #
    if not threshold and auto:
        lim = math.ceil(math.log(len(data), 10))
        if lim > 0:
            lim = lim if lim < 3 else 3
            threshold = [2**x for x in range(lim) if lim < 3]
        else:
            return(False)

    peaks = PeakDetection(data=data, threshold=threshold)
    signals = peaks.get_signals()
    stdFilter = peaks.get_stdfilter()
    avgFilter = peaks.get_avgFilter()

    if plot:
        plt.subplot(211)
        plt.plot(np.arange(1, len(y)+1), y)

        plt.plot(
            np.arange(1, len(y)+1),
            avgFilter,
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
