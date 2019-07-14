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

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle('Smoothed Z-Score', fontsize=16)

    #
    # data: with smoothed filter between upper and lower bound.
    #
    axs[0].plot(np.arange(1, len(data)+1), data)
    axs[0].plot(
        np.arange(1, len(data)+1),
        avg_filter,
        color='cyan',
        lw=2
    )
    axs[0].plot(
        np.arange(1, len(data)+1),
        avg_filter + threshold * std_filter,
        color='green',
        lw=2
    )
    axs[0].plot(
        np.arange(1, len(data)+1),
        avg_filter - threshold * std_filter,
        color='green',
        lw=2
    )
    axs[0].set_title('Data: with smoothed filter between bounds.')

    #
    # signal: associated signal to provided data.
    #
    axs[1].step(
        np.arange(1, len(data)+1),
        signals,
        color='red',
        lw=2
    )
    axs[1].set_title('Signal')

    #
    # save and plot
    #
    plt.savefig('{d}/peak_detection{suffix}'.format(
        d=directory,
        suffix=suffix
    ))

    if show:
        plt.show()
    else:
        plt.close()
