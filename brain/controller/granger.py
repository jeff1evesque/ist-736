#!/usr/bin/python

import numpy as np
from brain.algorithm.granger import granger as gr
from brain.view.plot import plot_bar


def granger(df, maxlag=1, directory='viz', suffix='', plot=True):
    '''

    implement granger test for causality.

    '''

    if suffix:
        suffix = '_{s}'.format(s=suffix)

    #
    # replace 'nan' with average value
    #
    # Note: this remediates statsmodels 'exog contains inf or nans' error.
    #
    df.ix[:,0] = [x if str(x) != 'nan'
        else np.nanmean(df.ix[:,0])
            for x in df.ix[:,0]]

    df.ix[:,1] = [x if str(x) != 'nan'
        else np.nanmean(df.ix[:,1])
            for x in df.ix[:,1]]

    result = [(k,v) for k,v in gr(df, maxlag=maxlag).items() if v]

    if plot:
        #
        # ssr based F test
        #
        plot_bar(
            labels = [x[0] for x in result],
            performance = [x[1][0]['ssr_ftest'][0] for x in result],
            directory=directory,
            filename='granger_ftest_statistic{s}'.format(s=suffix)
        )

        plot_bar(
            labels = [x[0] for x in result],
            performance = [x[1][0]['ssr_ftest'][1] for x in result],
            directory=directory,
            filename='granger_ftest_pvalue{s}'.format(s=suffix)
        )

        plot_bar(
            labels = [x[0] for x in result],
            performance = [x[1][0]['ssr_ftest'][2] for x in result],
            directory=directory,
            filename='granger_ftest_df_denom{s}'.format(s=suffix)
        )

        plot_bar(
            labels = [x[0] for x in result],
            performance = [x[1][0]['ssr_ftest'][3] for x in result],
            directory=directory,
            filename='granger_ftest_df_num{s}'.format(s=suffix)
        )

        #
        # ssr based chi2 test
        #
        plot_bar(
            labels = [x[0] for x in result],
            performance = [x[1][0]['ssr_chi2test'][0] for x in result],
            directory=directory,
            filename='granger_chitest_statistic{s}'.format(s=suffix)
        )

        plot_bar(
            labels = [x[0] for x in result],
            performance = [x[1][0]['ssr_chi2test'][1] for x in result],
            directory=directory,
            filename='granger_chitest_pvalue{s}'.format(s=suffix)
        )

        plot_bar(
            labels = [x[0] for x in result],
            performance = [x[1][0]['ssr_chi2test'][2] for x in result],
            directory=directory,
            filename='granger_chitest_df_denom{s}'.format(s=suffix)
        )

        #
        # likelihood ratio test
        #
        plot_bar(
            labels = [x[0] for x in result],
            performance = [x[1][0]['lrtest'][0] for x in result],
            directory=directory,
            filename='granger_ratiotest_statistic{s}'.format(s=suffix)
        )

        plot_bar(
            labels = [x[0] for x in result],
            performance = [x[1][0]['lrtest'][1] for x in result],
            directory=directory,
            filename='granger_ratiotest_pvalue{s}'.format(s=suffix)
        )

        plot_bar(
            labels = [x[0] for x in result],
            performance = [x[1][0]['lrtest'][2] for x in result],
            directory=directory,
            filename='granger_ratiotest_df_denom{s}'.format(s=suffix)
        )

        #
        # parameter F test
        #
        plot_bar(
            labels = [x[0] for x in result],
            performance = [x[1][0]['params_ftest'][0] for x in result],
            directory=directory,
            filename='granger_paramsftest_statistic{s}'.format(s=suffix)
        )

        plot_bar(
            labels = [x[0] for x in result],
            performance = [x[1][0]['params_ftest'][1] for x in result],
            directory=directory,
            filename='granger_paramsftest_pvalue{s}'.format(s=suffix)
        )

        plot_bar(
            labels = [x[0] for x in result],
            performance = [x[1][0]['params_ftest'][2] for x in result],
            directory=directory,
            filename='granger_paramsftest_df_denom{s}'.format(s=suffix)
        )
