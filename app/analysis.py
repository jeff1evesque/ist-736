#!/usr/bin/python

import os
import numpy as np
import pandas as pd
from datetime import datetime
from brain.view.plot import plot_bar
from brain.exploratory.sentiment import Sentiment
from brain.controller.classifier import classify
from brain.controller.timeseries import timeseries
from brain.controller.granger import granger
from brain.controller.peak_detection import peak_detection as signals


def analyze(
    data,
    df_quandl,
    screen_name,
    stopwords=[],
    directory='viz',
    directory_report='reports',
    sentiments = ['negative', 'neutral', 'positive'],
    classify_index='full_text',
    ts_index='value',
    analysis_ts=True,
    analysis_ts_sentiment=True,
    analysis_granger=True,
    analysis_classify=True
):
    '''

    general analysis on provided screen names.

    '''

    classify_results = {}
    ts_results = {}
    ts_results_sentiment = {}

    #
    # create directories
    #
    if not os.path.exists(directory_report):
        os.makedirs(directory_report)

    for i,sn in enumerate(screen_name):
        if not os.path.exists('{directory}/{sn}/granger'.format(
            directory=directory,
            sn=sn
        )):
            os.makedirs('{directory}/{sn}/granger'.format(
                directory=directory,
                sn=sn
            ))

    #
    # timeseries analysis: sentiment
    #
    for i,sn in enumerate(screen_name):
        # consistent datetime
        data[sn]['created_at'] = [datetime.strptime(
            x.split()[0],
            '%Y-%m-%d'
        ) for x in data[sn]['created_at']]

        # convert to string
        data[sn]['created_at'] = data[sn]['created_at'].astype(str)

        #
        # some screen_name text multiple times a day, yet quandl only provides
        #     daily prices.
        #
        data[sn] = data[sn].groupby([
            'created_at',
            'screen_name',
            'positive',
            'neutral',
            'negative'
        ]).agg({
            classify_index: lambda a: ''.join(map(str, a))
        }).reset_index()

        if analysis_ts_sentiment:
            for sentiment in sentiments:
                ts_results_sentiment[sn] = timeseries(
                    df=data[sn],
                    normalize_key=sentiment,
                    date_index='created_at',
                    directory='{directory}/{sn}'.format(
                        directory=directory,
                        sn=sn
                    ),
                    suffix=sentiment,
                    lstm_epochs=50
                )

                with open('{directory}/adf_{sn}_{sent}.txt'.format(
                    directory=directory_report,
                    sn=sn,
                    sent=sentiment
                ), 'w') as fp:
                    print(
                        ts_results_sentiment[sn]['arima']['adf'],
                        file=fp
                    )

    if analysis_ts_sentiment:
        s1 = [v['arima']['mse'] for k,v in ts_results_sentiment.items()]
        plot_bar(
            labels=screen_name,
            performance=s1,
            directory='{directory}'.format(directory=directory),
            filename='mse_overall_arima_sentiment.png',
            rotation=90
        )

        s2 = [v['lstm']['mse'][1] for k,v in ts_results_sentiment.items()]
        plot_bar(
            labels=screen_name,
            performance=s2,
            directory='{directory}'.format(directory=directory),
            filename='mse_overall_lstm_sentiment.png',
            rotation=90
        )

    #
    # preprocess: left join on twitter dataset(s).
    #
    for i,sn in enumerate(screen_name):
        # merge with consistent date format
        data[sn]['created_at'] = [datetime.strptime(
            x.split()[0],
            '%Y-%m-%d'
        ) for x in data[sn]['created_at']]

        # convert to string
        data[sn]['created_at'] = data[sn]['created_at'].astype(str)

        #
        # some screen_name text multiple times a day, yet quandl only provides
        #     daily prices.
        #
        data[sn] = data[sn].groupby([
            'created_at',
            'screen_name'
        ]).agg({
            classify_index: lambda a: ''.join(map(str, a))
        }).reset_index()

        data[sn] = data[sn].set_index('created_at').join(
            df_quandl.set_index('date'),
            how='left'
        ).reset_index()

        # column names: used below
        col_names = data[sn].columns.tolist()

        #
        # merge days (weekend, holidays) with no ticker value to previous day.
        #
        drop_indices = []
        for i,(idx,row) in enumerate(data[sn].iterrows()):
            if (i == 0 and pd.isnull(data[sn][ts_index][i])):
                drop_indices.append(i)

            elif (i > 0 and pd.isnull(data[sn][ts_index][i])):
                if not pd.isnull(data[sn][ts_index][i-1]):
                    for x in col_names:
                        if x == classify_index:
                            data[sn][classify_index][i] = '{previous} {current}'.format(
                                previous=data[sn][classify_index][i-1],
                                current=data[sn][classify_index][i]
                            )
                        else:
                            data[sn][x][i] = data[sn][x][i-1]

                    drop_indices.append(i-1)

        #
        # drop rows: rows with no tickers and empty classify_index.
        #
        drop_indices.extend(data[sn][data[sn][classify_index] == ''].index)
        data[sn] = data[sn].drop(data[sn].index[drop_indices]).reset_index()

        #
        # index data: conditionally apply z-score threshold.
        #
        signals = peak_detection(
            data=data[sn][ts_index],
            directory=directory,
            suffix=sn
        )

        # case 1: z-score index threshold determines trend
        if signals:
            signal_range = range(1, len(signals) + 1)
            signal_stream = {x: signals[x] for x in signal_range}

            data[sn]['trend'] = [-z
                if any([False if a < 0 else True for a in signal_stream[z]])
                else z
                for y in signal_stream[z] for z in signal_range]

        # case 2: relabel up (0) or down (1) based on previous index value
        else:
            data[sn]['trend'] = [0
                if data[sn][ts_index][i] > data[sn][ts_index].get(i-1, 0)
                else 1
                for i,x in enumerate(data[sn][ts_index])]

        # sentiment analysis
        s = Sentiment(data[sn], classify_index)
        data[sn] = pd.concat([s.vader_analysis(), data[sn]], axis=1)
        data[sn].replace('\s+', ' ', regex=True, inplace=True)

        #
        # granger causality
        #
        if analysis_granger:
            for sentiment in sentiments:
                granger(
                    data[sn][[ts_index, sentiment]],
                    maxlag=3,
                    directory='{directory}/{sn}/granger'.format(
                        directory=directory,
                        sn=sn
                    ),
                    suffix=sentiment
                )

        #
        # classify
        #
        if analysis_classify:
            classify_results[sn] = classify(
                data[sn],
                key_class='trend',
                key_text=classify_index,
                directory='{directory}/{sn}'.format(
                    directory=directory,
                    sn=sn
                ),
                top_words=25,
                stopwords=stopwords,
                k=500
            )

        #
        # timeseries analysis
        #
        if analysis_ts:
            ts_results[sn] = timeseries(
                df=data[sn],
                normalize_key=ts_index,
                date_index='created_at',
                directory='{directory}/{sn}'.format(directory=directory, sn=sn)
            )

            with open('{directory}/adf_{sn}.txt'.format(
                directory=directory_report,
                sn=sn
            ), 'w') as fp:
                print(ts_results[sn]['arima']['adf'], file=fp)

    #
    # ensembled scores
    #
    if analysis_classify:
        plot_bar(
            labels=screen_name,
            performance=[v[0] for k,v in classify_results.items()],
            directory='{directory}'.format(directory=directory),
            filename='accuracy_overall.png',
            rotation=90
        )

    if analysis_ts:
        plot_bar(
            labels=screen_name,
            performance=[v['arima']['mse'] for k,v in ts_results.items()],
            directory='{directory}'.format(directory=directory),
            filename='mse_overall_arima.png',
            rotation=90
        )

        plot_bar(
            labels=screen_name,
            performance=[v['lstm']['mse'][1] for k,v in ts_results.items()],
            directory='{directory}'.format(directory=directory),
            filename='mse_overall_lstm.png',
            rotation=90
        )
