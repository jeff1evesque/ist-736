#!/usr/bin/python

import os


def create_directory(
    screen_name,
    directory_granger='viz/granger',
    directory_lstm='viz/lstm',
    directory_arima='viz/arima',
    directory_class='viz/classification',
    directory_report='reports'
):
    '''

    create directories needed for analysis.

    '''

    directories = [
        directory_report,
        '{a}/stock'.format(a=directory_lstm),
        '{a}/stock'.format(a=directory_arima)
    ]
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
        for sn in screen_name:
            full_path = '{d}/{sn}'.format(d=d, sn=sn).split('/')
            if not os.path.exists(os.path.join(*full_path)):
                os.makedirs(os.path.join(*full_path))