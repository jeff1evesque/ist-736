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
    ts_index='value'
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

        #
        # index data: conditionally use z-score threshold to relabel index.
        #
        threshold = [0.5]
        signals = peak_detection(
            data=data[sn][ts_index],
            threshold=threshold,
            directory='{a}/{b}'.format(a=directory, b=sn),
            suffix=sn
        )

        #
        # case 1: z-score threshold determines trend index
        #
        if signals:
            signal_result = []
            for z in range(1, len(signals) + 1):
                signal = signals[z-1]
                for i,s in enumerate(signal):
                    if (len(signal_result) == 0 or len(signal_result) == i):
                        if s < 0:
                            signal_result.append(-z)
                        elif s > 0:
                            signal_result.append(z)
                        else:
                            signal_result.append(0)

                    elif (
                        i < len(signal_result) and
                        s < 0 and
                        s < signal_result[i]
                    ):
                        signal_result[i] = -z

                    elif (
                        i < len(signal_result) and
                        s > 0 and
                        s > signal_result[i]
                    ):
                        signal_result[i] = z

                    elif (
                        i < len(signal_result) and
                        s == 0 and
                        s > signal_result[i]
                    ):
                        signal_result[i] = 0

                    else:
                        print('Error ({f}): {m}.'.format(
                            f=this_file,
                            m='distorted signal_result shape'
                        ))

            # monotic: if all same values use non z-score.
            first = signal_result[0]
            if all(x == first for x in signal_result):
                data[sn]['trend'] = [0
                    if data[sn][ts_index].values[i] > data[sn][ts_index].get(i-1, 0)
                    else 1
                    for i,x in enumerate(data[sn][ts_index])]

            # not monotonic
            else:
                data[sn]['trend'] = signal_result

        #
        # case 2: previous index value determines trend index
        #
        else:
            data[sn]['trend'] = [0
                if data[sn][ts_index].values[i] > data[sn][ts_index].get(i-1, 0)
                else 1
                for i,x in enumerate(data[sn][ts_index])]

    return(data)
