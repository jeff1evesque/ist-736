#!/usr/bin/python

#
# this project requires the following packages:
#
#   pip update
#   pip install Twython Quandl wordcloud scikit-plot statsmodels patsy tensorflow seaborn h5py
#   pip install keras==2.1.2
#   pip install numpy==1.16.2
#

#
# nltk data
#
import nltk
for x in ['averaged_perceptron_tagger', 'stopwords']:
    nltk.download(x)

#
# register converters
#
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#
# general import
#
import re
import pandas as pd
from pathlib import Path
from datetime import date, datetime
import dateutil.relativedelta
from consumer.twitter import tweet_sn
from consumer.quandl import quandl
from app.exploratory import explore
from app.join_data import join_data
from app.analysis import analyze, analyze_ts
from app.create_directory import create_directory
from brain.utility.stopwords import stopwords, stopwords_topics
from config import (
    twitter_accounts as accounts,
    stock_codes,
    drop_cols,
    model_control as m,
    model_config as c,
    sentiments
)

#
# local variables
#
end_date = date.today()
start_date = end_date - dateutil.relativedelta.relativedelta(years=5)
stopwords.extend([x.lower() for x in accounts])
stopwords_topics.extend(stopwords)

#
# create directories
#
create_directory(
    screen_name=accounts,
    stock_codes=stock_codes,
    directory_lstm='viz/lstm_{a}'.format(a=c['lstm_epochs'])
)

#
# harvest tweets
#
data, start_date, end_date = tweet_sn(
    accounts,
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d')
)

#
# exploration: specific and overall tweets
#
if m['analysis_explore']:
    explore(
        data,
        accounts,
        stopwords=stopwords,
        stopwords_topics=stopwords_topics,
        directory='viz/exploratory'
    )

#
# harvest quandl
#
df_quandl = quandl(
    codes=stock_codes,
    start_date=start_date,
    end_date=end_date
)

#
# analysis
#
# @arima_auto_scale, only applied to sentiment timeseries analysis.
#
for x in df_quandl:
    #
    # local variables
    #
    sub_directory = '{b}--{c}'.format(
        b=x['database'].lower(),
        c=x['dataset'].lower()
    )

    #
    # join data: twitter and quandl
    #
    # Note: memory footprint reduced by removing unused columns.
    #
    df = join_data(
        data=data,
        df_quandl=x['data'],
        screen_name=accounts,
        drop_cols=drop_cols,
        sentiments=sentiments,
        classify_index=c['classify_index'],
        ts_index=c['ts_index']
    )[1]

    #
    # general analysis
    #
    if (
        m['analysis_granger'] or
        m['analysis_ts_stock'] or
        m['analysis_classify']
    ):
        analyze(
            df=df,
            df_quandl=x['data'],
            arima_auto_scale=c['=arima_auto_scale'],
            lstm_epochs=c['lstm_epochs'],
            lstm_dropout=c['lstm_dropout'],
            lstm_num_cells=c['lstm_num_cells'],
            lstm_units=c['lstm_units'],
            classify_chi2=c['classify_chi2'],
            classify_index=c['classify_index'],
            classify_threshold=c['classify_threshold'],
            sub_directory=sub_directory,
            directory_granger='viz/granger/{a}'.format(a=sub_directory),
            directory_lstm='viz/lstm_{a}'.format(a=c['lstm_epochs']),
            directory_arima='viz/arima',
            directory_class='viz/classification/{a}'.format(a=sub_directory),
            directory_report='reports/{a}'.format(a=x['dataset']),
            screen_name=accounts,
            stopwords=stopwords,
            ts_index=c['ts_index'],
            analysis_granger=analysis_granger,
            analysis_ts=analysis_ts_stock,
            analysis_classify=analysis_classify
        )

    else:
        break

#
# sentiment timeseries: analysis on twitter corpus for each financial analyst.
#
# Note: only last instance of joined dataframe (i.e. df) is needed, since the
#       twitter corpus is independent of the different stock index. Generally,
#       the intersection between twitter and different stock index (needed to
#       eliminate redundancies, such as repeated dates from twitter corpus)
#       will either not differ, or be an insignificant difference.
#
if m['analysis_ts_sentiment']:
    analyze_ts(
        df,
        accounts,
        arima_auto_scale=c['=arima_auto_scale'],
        lstm_epochs=c['lstm_epochs'],
        lstm_dropout=c['lstm_dropout'],
        lstm_num_cells=c['lstm_num_cells'],
        lstm_units=c['lstm_units'],
        directory_lstm='viz/lstm_{a}'.format(a=c['lstm_epochs']),
        directory_arima='viz/arima'
    )
