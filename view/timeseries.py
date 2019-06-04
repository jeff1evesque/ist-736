#!/usr/bin/python

import seaborn as sns
sns.set(style='darkgrid')

def plot_ts(
    data,
    xlab='dates',
    ylab='values',
    hue='variable',
    directory='viz',
    filename='ts',
    show=False
):
    '''

    plot confusion matrix.

    Note: multi-timeseries requires multiple columns melted.

    '''

    ax = sns.lineplot(
        x=xlab,
        y=ylab,
        hue='region',
        style='event',
        data=data
    )
    ax.set(xlabel=xlab, ylabel=ylab)
    plt.savefig('{d}/{f}'.format(d=directory, f=filename))

    if show:
        plt.show()
    else:
        plt.close()
