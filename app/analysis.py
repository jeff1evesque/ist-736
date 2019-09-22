#!/usr/bin/python

import os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from brain.view.plot import plot_bar
from brain.controller.classifier import classify
from brain.controller.timeseries import Timeseries
from brain.controller.granger import granger
from brain.controller.peak_detection import peak_detection
from app.join_data import join_data


def analyze(
    data,
    df_quandl,
    screen_name,
    stopwords=[],
    directory_granger='viz/granger',
    directory_lstm='viz/lstm',
    directory_arima='viz/arima',
    directory_class='viz/classification',
    directory_report='reports',
    arima_auto_scale=None,
    lstm_epochs=750,
    sentiments = ['negative', 'neutral', 'positive'],
    classify_index='full_text',
    classify_chi2=100,
    classify_threshold=[0.5],
    ts_index='value',
    analysis_granger=True,
    analysis_ts=True,
    analysis_ts_sentiment=True,
    analysis_classify=True
):
    '''

    general analysis on provided screen names.

    '''

    classify_results = {}
    ts_results = {}
    ts_results_sentiment = {}
    g = ['created_at', 'screen_name'] + sentiments
    drop_cols = [
        'compound',
        'retweet_count',
        'favorite_count',
        'user_mentions',
        'Short Volume'
    ]
    this_file = os.path.basename(__file__)
    directories = [
        directory_report,
        '{a}/stock'.format(a=directory_lstm),
        '{a}/sentiment'.format(a=directory_lstm),
        '{a}/stock'.format(a=directory_arima)
        '{a}/sentiment'.format(a=directory_arima)
    ]
    directories_sn = [directory_granger, directory_class]

    #
    # create directories
    #
    if not os.path.exists(directories):
        os.makedirs(directories)

    for d in directories_sn:
        for i,sn in enumerate(screen_name):
            full_path = '{d}/{sn}'.format(d=d, sn=sn).split('/')
            if not os.path.exists(os.path.join(*full_path)):
                os.makedirs(os.path.join(*full_path))

    #
    # join data: twitter and quandl
    #
    # Note: memory footprint reduced by removing unused columns.
    #
    joined_data, joined_data_agg = join_data(
        data=data,
        df_quandl=df_quandl,
        screen_name=screen_name,
        drop_cols=drop_cols,
        sentiments=sentiments,
        classify_index=classify_index,
        ts_index=ts_index
    )

    #
    # granger analysis:
    #
    # Note: requires vader sentiment scores.
    #
    if analysis_granger:
        initialized_data = joined_data_agg

        for i,sn in enumerate(screen_name):
            # merge with consistent date format
            initialized_data[sn]['created_at'] = [datetime.strptime(
                x,
                '%Y-%m-%d'
            ) for x in initialized_data[sn].index.tolist()]

            # convert to string
            initialized_data[sn]['created_at'] = initialized_data[sn]['created_at'].astype(str)

            #
            # granger causality
            #
            # Note: this requires the above quandl join.
            #
            for sent in sentiments:
                if all(x in initialized_data[sn] for x in [ts_index, sent]):
                    granger(
                        initialized_data[sn][[ts_index, sent]],
                        maxlag=4,
                        directory='{d}/{sn}'.format(
                            d=directory_granger,
                            sn=sn
                        ),
                        suffix=sent
                    )

    #
    # timeseries analysis: sentiment
    #
    if analysis_ts_sentiment:
        initialized_data = joined_data_agg

        for sn in screen_name:
            ts_results_sentiment[sn] = {}

            #
            # timeseries model on sentiment
            #
            for sent in sentiments:
                ts_results_sentiment[sn][sent] = {}

                if (
                    'created_at' == initialized_data[sn].index.name and
                    sent in initialized_data[sn]
                ):
                    ts_sentiment = Timeseries(
                        df=initialized_data[sn],
                        normalize_key=sent,
                        date_index=None,
                        directory_arima='{d}/{sn}'.format(
                            d=directory_arima,
                            sn=sn
                        ),
                        directory_lstm='{d}/{sn}'.format(
                            d=directory_lstm,
                            sn=sn
                        ),
                        suffix=sent,
                        arima_auto_scale=arima_auto_scale,
                        lstm_epochs=lstm_epochs,
                        lstm_dropout=0,
                        catch_grid_search=True
                    )

                    scores = ts_sentiment.get_model_scores()
                    ts_results_sentiment[sn][sent] = scores

                    if ('arima' in ts_results_sentiment[sn][sent]):
                        with open('{d}/adf_{sn}_{sent}.txt'.format(
                            d=directory_report,
                            sn=sn,
                            sent=sent
                        ), 'w') as fp:
                            print(ts_results_sentiment[sn][sent]['arima']['adf'], file=fp)

        #
        # mse plot: aggregated on overall sentiment for a given stock.
        #
        for sent in sentiments:
            if any(
                pd.notnull(k) and
                pd.notnull(v) and
                k == 'arima' and
                'mse' in v and
                pd.notnull(v['mse'])
                    for sn in screen_name
                        for k,v in ts_results_sentiment[sn][sent].items()
            ):
                plot_bar(
                    labels=[sn
                        for sn in screen_name
                            if sent in ts_results_sentiment[sn]
                            for k,v in ts_results_sentiment[sn][sent].items()
                                if k == 'arima'],
                    performance=[v['mse']
                        for sn in screen_name
                            if sent in ts_results_sentiment[sn]
                            for k,v in ts_results_sentiment[sn][sent].items()
                                if k == 'arima' and
                                    'mse' in v and
                                    pd.notnull(v['mse'])],
                    directory='{a}/stock'.format(directory_arima),
                    filename='mse_overall_arima_{sent}.png'.format(sent=sent),
                    rotation=60
                )

            if any(
                pd.notnull(k) and
                pd.notnull(v) and
                k == 'lstm' and
                'mse' in v and
                pd.notnull(v['mse'])
                    for sn in screen_name
                        for k,v in ts_results_sentiment[sn][sent].items()
            ):
                plot_bar(
                    labels=[sn
                        for sn in screen_name
                            if sent in ts_results_sentiment[sn]
                            for k,v in ts_results_sentiment[sn][sent].items()
                                if k == 'lstm'],
                    performance=[v['mse']
                        for sn in screen_name
                            if sent in ts_results_sentiment[sn]
                            for k,v in ts_results_sentiment[sn][sent].items()
                                if k == 'lstm' and
                                    'mse' in v and
                                    pd.notnull(v['mse'])],
                    directory='{a}/stock'.format(directory_lstm),
                    filename='mse_overall_lstm_{sent}.png'.format(sent=sent),
                    rotation=60
                )

    #
    # timeseries analysis: overall stock index/volume
    #
    if analysis_ts:
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
        ts_stock = Timeseries(
            df=df_quandl,
            normalize_key=ts_index,
            date_index='date',
            directory_arima='{a}/sentiment'.format(directory_arima),
            directory_lstm='{a}/sentiment'.format(directory_lstm),
            suffix=ts_index,
            arima_auto_scale=(50, 0.15),
            lstm_epochs=lstm_epochs,
            lstm_dropout=0
        )
        ts_results = ts_stock.get_model_scores()

        if 'arima' in ts_results:
            plot_bar(
                labels=['overall'],
                performance=ts_results['arima']['mse'],
                directory='{a}/sentiment'.format(directory_arima),
                filename='mse_overall_arima.png',
                rotation=90
            )

            with open('{d}/adf_{type}.txt'.format(
                d=directory_report,
                type=ts_index
            ), 'w') as fp:
                print(ts_results['arima']['adf'], file=fp)

        if 'lstm' in ts_results:
            plot_bar(
                labels=['overall'],
                performance=ts_results['lstm']['mse'],
                directory='{a}/sentiment'.format(directory_lstm),
                filename='mse_overall_lstm.png',
                rotation=90
            )

    #
    # classification analysis: twitter corpus (X), with stock index (y)
    #
    if analysis_classify:
        for i,sn in enumerate(screen_name):
            data = peak_detection(
                data=joined_data_agg[sn],
                ts_index=ts_index,
                directory='{a}/{b}'.format(a=directory_class, b=sn),
                threshold=classify_threshold
            )

            #
            # outlier class: remove class if training distribution is less than
            #     50%, or greater than 150% of all other class distribution(s).
            #
            counter = defaultdict(lambda :0)
            for k in data['trend']:
                counter[k] += 1

            if len(counter) > 2:
                for key, val in counter.items():
                    if all(val > 1.5 * v for k,v in counter.items() if v != val):
                        data.drop(
                            data[data['trend'] == key].index.values[0],
                            inplace=True
                        )

                for key, val in counter.items():
                    if all(val < 0.5 * v for k,v in counter.items() if v != val):
                        data.drop(
                            data[data['trend'] == key].index.values,
                            inplace=True
                        )

            #
            # sufficient data: analysis performed if adequate amount of data.
            #
            # Note: 3 classes are utilized as a base case, while an outlier class
            #       can reduce the default to 2 classes. Each additional threshold
            #       adds two additional classes.
            #
            if data.shape[0] > (3 + ((len(classify_threshold) - 1) * 2)) * 20:
                if all(x in data for x in ['trend', classify_index]):
                    #
                    # target vector: requires a minimum of 2 unique classes to
                    #     train a classifier.
                    #
                    if len(data['trend'].unique().tolist()) > 1:
                        classify_results[sn] = classify(
                            data,
                            key_class='trend',
                            key_text=classify_index,
                            directory='{d}/{sn}'.format(
                                d=directory_class,
                                sn=sn
                            ),
                            top_words=25,
                            stopwords=stopwords,
                            k=classify_chi2
                        )

            if any(
                pd.notnull(k) and
                pd.notnull(v) and
                isinstance(v, tuple) for k,v in classify_results.items()
            ):
                plot_bar(
                    labels=[k for k,v in classify_results.items() if pd.notnull(k)],
                    performance=[v[0] for k,v in classify_results.items()
                        if pd.notnull(v) and isinstance(v, tuple)],
                    directory=directory_class,
                    filename='accuracy_overall.png',
                    rotation=90
                )
