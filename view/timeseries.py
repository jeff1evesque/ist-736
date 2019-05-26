#!/usr/bin/python

import seaborn as sns
sns.set(style='darkgrid')

def plot_ts(
    data,
    xlab='dates',
    ylab='values',
    hue='variable'
    legend=True,
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

    if legend:
        plt.legend(legend, ncol=2, loc='upper right')

    plt.ylabel(ylab)
    plt.savefig('{d}/{f}'.format(d=directory, f=filename))

    if show:
        plt.show()
    else:
        plt.close()
