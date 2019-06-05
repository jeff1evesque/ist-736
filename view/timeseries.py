#!/usr/bin/python

import seaborn as sns
sns.set(style='darkgrid')
import matplotlib.pyplot as plt

def plot_ts(
    data,
    xlab='dates',
    ylab='values',
    directory='viz',
    filename='ts',
    hue=None,
    style=None,
    show=False
):
    '''

    plot confusion matrix.

    Note: multi-timeseries requires multiple columns melted.

    '''

    if hue or style:
        ax = sns.lineplot(
            data=data,
            x=xlab,
            y=ylab,
            hue=hue,
            style=style
        )
        ax.set(xlabel=xlab, ylabel=ylab)

    else:
        plt.plot(data[xlab], data[ylab])

    plt.savefig('{d}/{f}'.format(d=directory, f=filename))
    if show:
        plt.show()
    else:
        plt.close()
