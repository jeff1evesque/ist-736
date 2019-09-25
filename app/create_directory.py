#!/usr/bin/python

import os


def create_directory(
    screen_name,
    stock_codes,
    directory_granger='viz/granger',
    directory_lstm='viz/lstm',
    directory_arima='viz/arima',
    directory_class='viz/classification',
    directory_report='reports'
):
    '''

    create directories needed for analysis.

    @screen_name, list of financial analyst names.
    @stock_codes, must of a list of two element tuples, consisting of
        the quandl database, and stock name.

    '''

    #
    # define directories
    #
    codes = ['{b}--{c}'.format(
        b=x[0].lower(),
        c=x[1].lower()
    ) for x in stock_codes]

    directories = ['{a}/{b}'.format(
        a=directory_report,
        b=x[1]
    ) for x in stock_codes])

    directories.extend(['{a}/stock/{b}'.format(
        a=directory_arima,
        b=b
    ) for b in codes])

    directories.extend(['{a}/stock/{b}'.format(
        a=directory_lstm,
        b=b
    ) for b in codes])

    directories_sn = [
        directory_granger,
        directory_class,
        '{a}/sentiment'.format(a=directory_lstm),
        '{a}/sentiment'.format(a=directory_arima)
    ]

    #
    # create directories
    #
    for d in directories:
        full_path = d.split('/')
        if not os.path.exists(os.path.join(*full_path)):
            os.makedirs(os.path.join(*full_path))

    for d in directories_sn:
        for x in screen_name:
            full_path = '{d}/{x}'.format(d=d, x=x).split('/')
            if not os.path.exists(os.path.join(*full_path)):
                os.makedirs(os.path.join(*full_path))
