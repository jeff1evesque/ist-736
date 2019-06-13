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
    rotation=0,
    show=False,
    xticks=True
):
    '''

    plot confusion matrix.

    Note: multi-timeseries requires multiple columns melted.

    '''

    ax = sns.lineplot(
        data=data,
        x=xlab,
        y=ylab,
        hue=hue,
        style=style
    )
    ax.set(xlabel=xlab, ylabel=ylab)

    plt.xticks(rotation=rotation)

    if not xticks:
        plt.xticks([])

    plt.tight_layout()
    plt.savefig('{d}/{f}'.format(d=directory, f=filename))

    if show:
        plt.show()
    else:
        plt.close()
