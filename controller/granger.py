#!/usr/bin/python

from algorithm.controller import granger as gr
from view.plot import plot_bar


def granger(df, maxlag=1, directory='viz', , suffix='', plot=True):
    '''

    implement granger test for causality.

    '''

    result = gr(df, maxlag=maxlag)

    if plot:
        plot_bar(
            labels = [*result],
            performance = [v[0][0] for k,v in result.items()],
            directory=directory,
            filename='granger_test_statistic{s}'.format(s=suffix)
        )

        plot_bar(
            labels = [*result],
            performance = [v[0][1] for k,v in result.items()],
            directory=directory,
            filename='granger_p_value{s}'.format(s=suffix)
        )

        with open('reports/ols_{s}.txt'.format(s=suffix), 'w') as fp:
            print([v[1] for k,v in result.items()], file=fp)
