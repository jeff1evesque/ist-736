#!/usr/bin/python

import math
from brain.model.peak_detection import peak_detection as pk_detect


def peak_detection(
    data,
    ts_index,
    directory='viz',
    plot=True,
    threshold=None,
    auto=True
):
    '''

    implement z-score peak detection, with default behavior if possible.

    '''

    #
    # index data: conditionally use z-score threshold to relabel index.
    #
    signals = pk_detect(
        data=data[ts_index],
        threshold=threshold,
        directory=directory,
        plot=plot
    )

    #
    # case 1: z-score threshold determines trend index
    #
    if signals:
        signal_result = []
        for z in range(1, len(signals) + 1):
            signal = signals[z-1]
            for i,s in enumerate(signal):
                if (len(signal_result) == 0 or len(signal_result) == i):
                    if s < 0:
                        signal_result.append(-z)
                    elif s > 0:
                        signal_result.append(z)
                    else:
                        signal_result.append(0)

                elif (
                    i < len(signal_result) and
                    s < 0 and
                    s < signal_result[i]
                ):
                    signal_result[i] = -z

                elif (
                    i < len(signal_result) and
                    s > 0 and
                    s > signal_result[i]
                ):
                    signal_result[i] = z

                elif (
                    i < len(signal_result) and
                    s == 0 and
                    s > signal_result[i]
                ):
                    signal_result[i] = 0

                else:
                    print('Error ({f}): {m}.'.format(
                        f=this_file,
                        m='distorted signal_result shape'
                    ))

        # monotic: if all same values use non z-score.
        first = signal_result[0]
        if all(x == first for x in signal_result):
            data['trend'] = [0
                if data[ts_index].values[i] > data[ts_index].get(i-1, 0)
                else 1
                for i,x in enumerate(data[ts_index])]

        # not monotonic
        else:
            data['trend'] = signal_result

    #
    # case 2: previous index value determines trend index
    #
    else:
        data['trend'] = [0
            if data[ts_index].values[i] > data[ts_index].get(i-1, 0)
            else 1
            for i,x in enumerate(data[ts_index])]

    return(data)
