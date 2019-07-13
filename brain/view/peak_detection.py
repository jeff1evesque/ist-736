#!/usr/bin/python

import numpy as np
import seaborn as sns
sns.set(style='darkgrid')
import matplotlib.pyplot as plt

def peak_detection(
    data,
    threshold,
    signals,
    std_filter,
    avg_filter,
    directory='viz',
    suffix='',
    show=False
):
    '''

    plot z-score peak detection.

    '''

    if suffix:
        suffix = '_{a}'.format(a=suffix)

    plt.subplot(211)
    plt.plot(np.arange(1, len(data)+1), data)

    plt.plot(
        np.arange(1, len(data)+1),
        avg_filter,
        color='cyan',
        lw=2
    )

    plt.plot(
        np.arange(1, len(data)+1),
        avg_filter + threshold * std_filter,
        color='green',
        lw=2
    )

    plt.plot(
        np.arange(1, len(data)+1),
        avg_filter - threshold * std_filter,
        color='green',
        lw=2
    )

    plt.subplot(212)
    plt.step(
        np.arange(1, len(data)+1),
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
