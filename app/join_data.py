#!/usr/bin/python

import os
import pandas as pd
from datetime import datetime
from app.utility.merge_records import merge_records
from app.utility.merge_sentiment import merge_sentiment
from app.utility.drop_columns import drop_columns


def join_data(
    data,
    df_quandl,
    screen_name,
    drop_cols=None,
    drop_cols_regex=['^Unnamed'],
    sentiments = ['negative', 'neutral', 'positive'],
    classify_index='full_text',
    ts_index='value',
    threshold = [0.5]
):
    '''

    initialize twitter and quandl dataset(s).

    '''

    data_agg = {}
    group_ts = ['created_at', 'screen_name']
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

        if 'date' in df_quandl:
            df_quandl.set_index('date', inplace=True)
            df_quandl.index.name = 'created_at'

        #
        # aggregate records: combine records by 'classify_index'
        #
        data_agg[sn] = data[sn]
        data_agg[sn] = data[sn].groupby(group_ts).agg({
                classify_index: lambda a: ' '.join(map(str, a))
            }).reset_index()

        #
        # merge dataset: twitter and quandl
        #
        for x in df_quandl.columns:
            if x in data[sn]:
                data[sn].drop([x], axis = 1, inplace=True)
            if x in data_agg[sn]:
                data_agg[sn].drop([x], axis = 1, inplace=True)

        data[sn] = data[sn].merge(df_quandl, on='created_at', how='left')
        data_agg[sn] = data_agg[sn].merge(df_quandl, on='created_at', how='left')

        #
        # merge days (weekend, holidays) with no ticker value to previous day.
        #
        data[sn] = merge_records(data[sn], ts_index, classify_index)
        data_agg[sn] = merge_records(data_agg[sn], ts_index, classify_index)

        #
        # ensure no duplicate columns
        #
        # Note: reset_index workaround implemented for below concat
        #
        #     - https://stackoverflow.com/a/47991762
        #
        duplicates = sentiments + ['compound']
        for x in duplicates:
            if x in data[sn]:
                data[sn].drop(x, inplace=True, axis=1)
            if x in data_agg[sn]:
                data_agg[sn].drop(x, inplace=True, axis=1)

        data[sn]['created_at'] = data[sn].index.values
        data_agg[sn]['created_at'] = data_agg[sn].index.values

        data[sn].reset_index(drop=True, inplace=True)
        data_agg[sn].reset_index(drop=True, inplace=True)

        #
        # merge sentiment scores
        #
        data[sn] = merge_sentiment(
            data[sn],
            classify_index,
            df_index='created_at'
        )

        data_agg[sn] = merge_sentiment(
            data_agg[sn],
            classify_index,
            df_index='created_at'
        )

        #
        # drop columns: remove unused columns to reduce memory footprint
        #
        data[sn] = drop_columns(data[sn], drop_cols, drop_cols_regex)
        data_agg[sn] = drop_columns(data_agg[sn], drop_cols, drop_cols_regex)

    return(data, data_agg)
