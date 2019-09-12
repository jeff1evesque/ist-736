#!/usr/bin/python

import os
import pandas as pd
from datetime import datetime
from brain.controller.peak_detection import peak_detection


def join_data(
    data,
    df_quandl,
    screen_name,
    directory='viz',
    directory_report='reports',
    sentiments = ['negative', 'neutral', 'positive'],
    classify_index='full_text',
    ts_index='value',
    threshold = [0.5]
):
    '''

    initialize twitter and quandl dataset(s).

    '''

    classify_results = {}
    ts_results = {}
    ts_results_sentiment = {}
    g = ['created_at', 'screen_name'] + sentiments
    this_file = os.path.basename(__file__)

    #
    # preprocess: left join on twitter dataset(s).
    #
    for i,sn in enumerate(screen_name):

        if 'created_at' in data[sn]:
            # merge with consistent date format
            data[sn]['created_at'] = [datetime.strptime(
                x.split()[0],
                '%Y-%m-%d'
            ) for x in data[sn]['created_at']]

            # convert to string
            data[sn]['created_at'] = data[sn]['created_at'].astype(str)

            # set index
            data[sn].set_index('created_at', inplace=True)
            data[sn].index.name = 'created_at'

            #
            # some screen_name text multiple times a day, yet quandl only provides
            #     daily prices.
            #
            data[sn] = data[sn].groupby(g).agg({
                classify_index: lambda a: ''.join(map(str, a))
            }).reset_index()

            # set index
            data[sn].set_index('created_at', inplace=True)
            data[sn].index.name = 'created_at'

        if 'date' in df_quandl:
            df_quandl.set_index('date', inplace=True)
            df_quandl.index.name = 'created_at'

        #
        # merge dataset: twitter and quandl
        #
        data[sn] = data[sn].merge(df_quandl, on='created_at', how='left')

        # column names: used below
        col_names = data[sn].columns.tolist()

        #
        # merge days (weekend, holidays) with no ticker value to previous day.
        #
        drop_indices = []
        for i,(idx,row) in enumerate(data[sn].iterrows()):
            if (i == 0 and pd.isnull(data[sn][ts_index].values[i])):
                drop_indices.append(i)

            elif (i > 0 and pd.isnull(data[sn][ts_index].values[i])):
                if not pd.isnull(data[sn][ts_index].values[i-1]):
                    for x in col_names:
                        if x == classify_index:
                            data[sn][classify_index].replace(
                                i,
                                '{previous} {current}'.format(
                                    previous=data[sn][classify_index].values[i-1],
                                    current=data[sn][classify_index].values[i]
                                )
                            )
                        else:
                            data[sn][x].replace(i, data[sn][x].values[i-1])

                    drop_indices.append(i-1)

        #
        # drop rows: rows with no tickers and empty classify_index.
        #
        drop_indices.extend(data[sn][data[sn][classify_index] == ''].index)
        data[sn] = data[sn].drop(data[sn].index[drop_indices]).reset_index()

    return(data)
