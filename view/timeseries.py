#!/usr/bin/python

import seaborn as sns
sns.set(style='darkgrid')

def plot_ts(
    data,
    xlab='Dates',
    ylab='Values',
    directory='viz',
    file_suffix='text'
):
    '''

    plot confusion matrix.

    '''

    sns.lineplot(
        x='timepoint',
        y='signal',
        hue='region',
        style='event',
        data=fmri
    )
