#!/usr/bin/python

import math
from brain.algorithm.peak_detection import PeakDetection
from brain.view.peak_detection import peak_detection as plot_pd


def peak_detection(
    data,
    directory='viz',
    suffix='',
    plot=True,
    threshold=None,
    auto=True
):
    '''

    implement z-score peak detection, with default behavior if possible.

    '''

    #
    # threshold: autogenerate if not provided, cancel if not enough data.
    #
    if not threshold and auto:
        lim = math.ceil(math.log(len(data), 10))
        if lim > 0:
            lim = lim if lim < 3 else 3
            threshold = [2**x for x in range(lim)]
        else:
            return(False)

    #
    # initialize: instantiate and create z-score model
    #
    peaks = PeakDetection(data=data, threshold=threshold)
    data = peaks.get_data()
    signals = peaks.get_signals()
    std_filter = peaks.get_std_filter()
    avg_filter = peaks.get_avg_filter()

    if plot:
        for i,x in enumerate(threshold):
            plot_pd(
                data=data,
                threshold=x,
                signals=signals[i],
                std_filter=std_filter[i],
                avg_filter=avg_filter[i],
                directory=directory,
                suffix=i
            )

    return(signals)
