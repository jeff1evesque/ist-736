#!/usr/bin/python

#
# this project requires the following packages:
#
#   pip update
#   pip install Twython Quandl wordcloud scikit-plot statsmodels patsy tensorflow seaborn
#   pip install keras==2.1.2
#   pip install numpy==1.16.2
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

#
# local variables
#
screen_name = [
    'jimcramer',
    'ReformedBroker',
    'TheStalwart',
    'LizAnnSonders',
    'SJosephBurns'
]
codes = [
    ('BATS', 'BATS_AAPL'),
##    ('BATS', 'BATS_AMZN'),
##    ('BATS', 'BATS_GOOGL'),
##    ('BATS', 'BATS_MMT'),
##    ('BATS', 'BATS_NFLX'),
##    ('CHRIS', 'CBOE_VX1'),
##    ('NASDAQOMX', 'COMP-NASDAQ'),
##    ('FINRA', 'FNYX_MMM'),
##    ('FINRA', 'FNSQ_SPY'),
##    ('FINRA', 'FNYX_QQQ'),
##    ('EIA', 'PET_RWTC_D'),
##    ('WFC', 'PR_CON_15YFIXED_IR'),
##    ('WFC', 'PR_CON_30YFIXED_APR')
]
end_date = date.today()
start_date = end_date - dateutil.relativedelta.relativedelta(years=5)
sentiments = ['negative', 'neutral', 'positive']
classify_index = 'full_text'
ts_index = 'value'

arima_auto_scale = None
lstm_epochs = 750
classify_threshold = [0.5]
classify_chi2 = 100

stopwords.extend([x.lower() for x in screen_name])
stopwords_topics.extend(stopwords)

drop_cols = [
    'compound',
    'retweet_count',
    'favorite_count',
    'user_mentions',
    'Short Volume'
]

#
# create directories
#
create_directory(
    screen_name,
    directory_lstm='viz/lstm_{a}'.format(a=lstm_epochs)
)

#
# harvest tweets
#
data, start_date, end_date = tweet_sn(
    screen_name,
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d')
)

#
# exploration: specific and overall tweets
#
###explore(
###    data,
###    screen_name,
###    stopwords=stopwords,
###    stopwords_topics=stopwords_topics,
###    directory='viz/exploratory'
###)

#
# harvest quandl
#
df_quandl = quandl(
    codes=codes,
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
        screen_name=screen_name,
        drop_cols=drop_cols,
        sentiments=sentiments,
        classify_index=classify_index,
        ts_index=ts_index
    )[1]

    #
    # general analysis
    #
    analyze(
        df=df,
        df_quandl=x['data'],
        arima_auto_scale=arima_auto_scale,
        lstm_epochs=lstm_epochs,
        classify_threshold=classify_threshold,
        directory_granger='viz/granger/{a}'.format(a=sub_directory),
        directory_lstm='viz/lstm_{a}/{b}'.format(
            a=lstm_epochs,
            b=sub_directory
        ),
        directory_arima='viz/arima/{a}'.format(a=sub_directory),
        directory_class='viz/classification/{a}'.format(a=sub_directory),
        directory_report='reports/{a}'.format(a=x['dataset']),
        screen_name=screen_name,
        stopwords=stopwords,
        classify_index=classify_index,
        ts_index=ts_index,
        analysis_granger=False,
        analysis_ts=False,
        analysis_classify=False
    )

#
# sentiment timeseries: analysis on twitter corpus for each financial analyst.
#
# Note: only last instance of joined dataframe (i.e. df) is needed, since stock
#       data is always a superset of twitter corpus.
#
analyze_ts(
    df,
    screen_name,
    arima_auto_scale=arima_auto_scale,
    lstm_epochs=lstm_epochs,
    directory_lstm='viz/lstm_{a}'.format(a=lstm_epochs),
    directory_arima='viz/arima'
)
