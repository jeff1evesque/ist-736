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
from app.join_data import join_data


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
    g = ['created_at', 'screen_name'] + sentiments
    this_file = os.path.basename(__file__)

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
    # join data: twitter and quandl
    #
    joined_data, joined_data_agg = join_data(
        data=data,
        df_quandl=df_quandl,
        screen_name=screen_name,
        directory=directory,
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
        initialized_data = joined_data

        for i,sn in enumerate(screen_name):
            # merge with consistent date format
            initialized_data[sn]['created_at'] = [datetime.strptime(
                x.split()[0],
                '%Y-%m-%d'
            ) for x in initialized_data[sn]['created_at']]

            # convert to string
            initialized_data[sn]['created_at'] = initialized_data[sn]['created_at'].astype(str)

            #
            # sentiment scores
            #
            s = Sentiment(initialized_data[sn], classify_index)
            initialized_data[sn] = pd.concat([
                s.vader_analysis(),
                initialized_data[sn]
             ], axis=1)
            initialized_data[sn].replace('\s+', ' ', regex=True, inplace=True)

            #
            # granger causality
            #
            # Note: this requires the above quandl join.
            #
            for sentiment in sentiments:
                if all(x in initialized_data[sn] for x in [ts_index, sentiment]):
                    granger(
                        initialized_data[sn][[ts_index, sentiment]],
                        maxlag=3,
                        directory='{directory}/{sn}/granger'.format(
                            directory=directory,
                            sn=sn
                        ),
                        suffix=sentiment
                    )

    #
    # timeseries analysis: sentiment
    #
    if analysis_ts_sentiment:
        for i,sn in enumerate(screen_name):
            #
            # sentiment scores
            #
            s = Sentiment(joined_data_agg[sn], classify_index)
            joined_data_agg[sn] = pd.concat([
                s.vader_analysis(),
                joined_data_agg[sn]
             ], axis=1)
            joined_data_agg[sn].replace('\s+', ' ', regex=True, inplace=True)

            #
            # timeseries model on sentiment
            #
            for sentiment in sentiments:
                if all(x in joined_data_agg[sn] for x in [sentiment, 'created_at']):
                    ts_sentiment = Timeseries(
                        df=joined_data_agg[sn],
                        normalize_key=sentiment,
                        date_index=None,
                        directory='{directory}/{sn}'.format(
                            directory=directory,
                            sn=sn
                        ),
                        suffix=sentiment,
                        lstm_epochs=7500,
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
            directory='{directory}'.format(directory=directory),
            suffix=ts_index,
            lstm_epochs=7500,
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

    #
    # classification analysis: twitter corpus (X), with stock index (y)
    #
    if analysis_classify:
        chi2 = 100
        classify_threshold = [0.5]

        for i,sn in enumerate(screen_name):
            data = peak_detection(
                data=joined_data[sn],
                ts_index=ts_index,
                directory='{a}/{b}'.format(a=directory, b=sn),
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
                    if all(val < 0.5 * v for k,v in counter.items() if v != val):
                        data.drop(
                            data[data['trend'] == key].index.values,
                            inplace=True
                        )
                        break

                    elif all(val > 1.5 * v for k,v in counter.items() if v != val):
                        data.drop(
                            data[data['trend'] == key].index.values,
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
                            directory='{directory}/{sn}'.format(
                                directory=directory,
                                sn=sn
                            ),
                            top_words=25,
                            stopwords=stopwords,
                            k=chi2
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
                    directory='{directory}'.format(directory=directory),
                    filename='accuracy_overall.png',
                    rotation=90
                )
