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
from app.analysis import analyze
from app.exploratory import explore
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
##    ('BATS', 'BATS_AAPL'),
##    ('BATS', 'BATS_AMZN'),
##    ('BATS', 'BATS_GOOGL'),
##    ('BATS', 'BATS_MMT'),
##    ('BATS', 'BATS_NFLX'),
##    ('CHRIS', 'CBOE_VX1'),
##    ('NASDAQOMX', 'COMP-NASDAQ'),
##    ('FINRA', 'FNYX_MMM'),
    ('FINRA', 'FNSQ_SPY'),
##    ('FINRA', 'FNYX_QQQ'),
##    ('EIA', 'PET_RWTC_D'),
##    ('WFC', 'PR_CON_15YFIXED_IR'),
##    ('WFC', 'PR_CON_30YFIXED_APR')
]
end_date = date.today()
start_date = end_date - dateutil.relativedelta.relativedelta(years=5)

stopwords.extend([x.lower() for x in screen_name])
stopwords_topics.extend(stopwords)

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
arima_auto_scale=None
lstm_epochs=750
classify_threshold=[0.5]
classify_chi2=100

for x in df_quandl:
    analyze(
        data=data,
        df_quandl=x['data'],
        arima_auto_scale=arima_auto_scale,
        lstm_epochs=lstm_epochs,
        classify_threshold=classify_threshold,
        directory_granger='viz/granger/{b}--{c}'.format(
            b=x['database'].lower(),
            c=x['dataset'].lower()
        ),
        directory_lstm='viz/lstm_{a}/{b}--{c}'.format(
            a=lstm_epochs,
            b=x['database'].lower(),
            c=x['dataset'].lower()
        ),
        directory_arima='viz/arima/{b}--{c}'.format(
            b=x['database'].lower(),
            c=x['dataset'].lower()
        ),
        directory_class='viz/classification/{b}--{c}'.format(
            b=x['database'].lower(),
            c=x['dataset'].lower()
        ),
        directory_report='reports/{a}/{b}'.format(
            a=lstm_epochs,
            b=x['dataset']
        ),
        screen_name=screen_name,
        stopwords=stopwords,
        analysis_granger=True,
        analysis_ts=False,
        analysis_ts_sentiment=False,
        analysis_classify=False
    )
