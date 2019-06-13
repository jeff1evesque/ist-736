#!/usr/bin/python

from statsmodels.tsa.stattools import grangercausalitytests


def granger(df, maxlag=1):
    '''

    implement granger test for causality.

    null hypothesis: the series in the second column, does not granger cause
        the series in the first column.

    @df, test time series in the second column Granger causes the time series
        in the first column.

    '''

    return(grangercausalitytests(df, maxlag=maxlag))
