#!/usr/bin/python

import os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from brain.view.plot import plot_bar
from brain.exploratory.sentiment import Sentiment
from brain.controller.classifier import classify
from brain.controller.timeseries import Timeseries
from brain.controller.granger import granger
from brain.controller.peak_detection import peak_detection


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
    analysis_ts=False,
    analysis_ts_sentiment=False,
    analysis_granger=True,
    analysis_classify=False
):
    '''

    general analysis on provided screen names.

    '''

    classify_results = {}
    ts_results = {}
    ts_results_sentiment = {}
    g = ['created_at', 'screen_name'] + sentiments
    this_file = os.path.basename(__file__)

    #
    # create directories
    #
    if not os.path.exists(directory_report):
        os.makedirs(directory_report)

    for i,sn in enumerate(screen_name):
        #
        # rename column: prevent collision with 'reset_index'
        #
        if 'index' in data[sn]:
            data[sn].rename(columns={'index': 'index_val'}, inplace=True)

        # create directories
        if not os.path.exists('{directory}/{sn}/granger'.format(
            directory=directory,
            sn=sn
        )):
            os.makedirs('{directory}/{sn}/granger'.format(
                directory=directory,
                sn=sn
            ))

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
        data[sn] = data[sn].groupby(g).agg({
            classify_index: lambda a: ''.join(map(str, a))
        }).reset_index()

        #
        # timeseries analysis: sentiment
        #
        if analysis_ts_sentiment:
            #
            # sentiment scores
            #
            s = Sentiment(data[sn], classify_index)
            data[sn] = pd.concat([s.vader_analysis(), data[sn]], axis=1)
            data[sn].replace('\s+', ' ', regex=True, inplace=True)

            #
            # timeseries model on sentiment
            #
            for sentiment in sentiments:
                if all(x in data[sn] for x in [sentiment, 'created_at']):
                    ts_sentiment = Timeseries(
                        df=data[sn],
                        normalize_key=sentiment,
                        date_index='created_at',
                        directory='{directory}/{sn}'.format(
                            directory=directory,
                            sn=sn
                        ),
                        suffix=sentiment,
                        lstm_epochs=1500,
                        lstm_dropout=0,
                        catch_grid_search=True
                    )
                    ts_results_sentiment[sn] = ts_sentiment.get_model_scores()

                    if 'arima' in ts_results_sentiment[sn]:
                        with open('{directory}/adf_{sn}_{sent}.txt'.format(
                            directory=directory_report,
                            sn=sn,
                            sent=sentiment
                        ), 'w') as fp:
                            print(
                                ts_results_sentiment[sn]['arima']['adf'],
                                file=fp
                            )

    #
    # plot sentiment scores
    #
    if analysis_ts_sentiment:
        if any(
            pd.notnull(k) and
            pd.notnull(v) and
            'arima' in v for k,v in ts_results_sentiment.items()
        ):
            plot_bar(
                labels=[k for k,v in ts_results_sentiment.items() if 'arima' in v],
                performance=[v['arima']['mse']
                    for k,v in ts_results_sentiment.items() if 'arima' in v],
                directory='{directory}'.format(directory=directory),
                filename='mse_overall_arima_sentiment.png',
                rotation=90
            )

        if any(
            pd.notnull(k) and
            pd.notnull(v) and
            'lstm' in v for k,v in ts_results_sentiment.items()
        ):
            plot_bar(
                labels=[k for k,v in ts_results_sentiment.items() if 'lstm' in v],
                performance=[v['lstm']['mse']
                    for k,v in ts_results_sentiment.items() if 'lstm' in v],
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
        data[sn] = data[sn].groupby(g).agg({
            classify_index: lambda a: ''.join(map(str, a))
        }).reset_index()

        data[sn] = data[sn].set_index('created_at').join(
            df_quandl.set_index('date'),
            how='left'
        ).reset_index()

        #
        # granger causality
        #
        # Note: this requires the above quandl join.
        #
        if analysis_granger:
            for sentiment in sentiments:
                if all(x in data[sn] for x in [ts_index, sentiment]):
                    granger(
                        data[sn][[ts_index, sentiment]],
                        maxlag=3,
                        directory='{directory}/{sn}/granger'.format(
                            directory=directory,
                            sn=sn
                        ),
                        suffix=sentiment
                    )

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
                    if data[sn][ts_index][i] > data[sn][ts_index].get(i-1, 0)
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
                if data[sn][ts_index][i] > data[sn][ts_index].get(i-1, 0)
                else 1
                for i,x in enumerate(data[sn][ts_index])]

        #
        # timeseries analysis: if dataset not 50 elements or more (x), use the
        #     default (p,q,d) range. Otherwise, the grid search (p,q,d) range
        #     is determined by 0.15x:
        #
        #     (p,q,d) = (range(0, 0.15x), range(0, 0.15x), range(0, 0.15x))
        #
        # Note: if arima does not converge, or the adf test does not satisfy
        #       p < 0.05, the corresponding model is thrown out.
        #
        if analysis_ts:
            ts_stock = Timeseries(
                df=data[sn],
                normalize_key=ts_index,
                date_index='created_at',
                directory='{directory}/{sn}'.format(directory=directory, sn=sn),
                suffix=ts_index,
                lstm_epochs=1500,
                lstm_dropout=0,
                auto_scale=(50, 0.15)
            )
            ts_results[sn] = ts_stock.get_model_scores()

            if 'arima' in ts_results[sn]:
                with open('{directory}/adf_{sn}_{type}.txt'.format(
                    directory=directory_report,
                    sn=sn,
                    type=ts_index
                ), 'w') as fp:
                    print(ts_results[sn]['arima']['adf'], file=fp)

        #
        # outlier class: remove class if training distribution is less than
        #     50%, or greater than 150% of all other class distribution(s).
        #
        counter = defaultdict(lambda :0)
        for k in data[sn]['trend']:
            counter[k] += 1

        if len(counter) > 2:
            for key, val in counter.items():
                if all(val < 0.5 * v for k,v in counter.items() if v != val):
                    data[sn].drop(
                        data[sn][data[sn]['trend'] == key].index.values,
                        inplace=True
                    )
                    break

                elif all(val > 1.5 * v for k,v in counter.items() if v != val):
                    data[sn].drop(
                        data[sn][data[sn]['trend'] == key].index,values,
                        inplace=True
                    )
                    break

        #
        # sufficient data: analysis performed if adequate amount of data.
        #
        # Note: 3 classes are utilized as a base case, while an outlier class
        #       can reduce the default to 2 classes. Each additional threshold
        #       adds two additional classes.
        #
        if data[sn].shape[0] > (3 + ((len(threshold) - 1) * 2)) * 20:
            #
            # classify
            #
            if analysis_classify:
                if all(x in data[sn] for x in ['trend', classify_index]):
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
                        k=750
                    )

    #
    # ensembled scores
    #
    if analysis_classify:
        if any(
            pd.notnull(k) and
            pd.notnull(v) and
            isinstance(v, tuple) for k,v in classify_results.items()
        ):
            plot_bar(
                labels=[k for k,v in classify_results.items() if pd.notnull(k)],
                performance=[v[0] for k,v in classify_results.items()
                    if pd.notnull(v) and isinstance(v, tuple)],
                directory='{directory}'.format(directory=directory),
                filename='accuracy_overall.png',
                rotation=90
            )

    if analysis_ts:
        if any(
            pd.notnull(k) and
            pd.notnull(v) and
            'arima' in v for k,v in ts_results.items()
        ):
            plot_bar(
                labels=[k for k,v in ts_results.items() if 'arima' in v],
                performance=[v['arima']['mse']
                    for k,v in ts_results.items() if 'arima' in v],
                directory='{directory}'.format(directory=directory),
                filename='mse_overall_arima.png',
                rotation=90
            )

        if any(
            pd.notnull(k) and
            pd.notnull(v) and
            'lstm' in v for k,v in ts_results.items()
        ):
            plot_bar(
                labels=[k for k,v in ts_results.items() if 'lstm' in v],
                performance=[v['lstm']['mse']
                    for k,v in ts_results.items() if 'lstm' in v],
                directory='{directory}'.format(directory=directory),
                filename='mse_overall_lstm.png',
                rotation=90
            )

