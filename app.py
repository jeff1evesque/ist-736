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
from datetime import datetime
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
    ('CHRIS', 'CBOE_VX1'),
    ('NASDAQOMX', 'COMP-NASDAQ'),
    ('FINRA', 'FNYX_QQQ')
]
start_date = datetime(3000, 12, 25)
end_date = datetime(1000, 12, 25)
stopwords.extend([x.lower() for x in screen_name])
stopwords_topics.extend(stopwords)

#
# harvest tweets
#
data, start_date, end_date = tweet_sn(
    screen_name,
    start_date,
    end_date
)

#
# exploration: specific and overall tweets
#
explore(
    data,
    screen_name,
    stopwords=stopwords,
    stopwords_topics=stopwords_topics
)

#
# harvest quandl
#
df_quandl = quandl(
    codes=codes,
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d')
)

#
# analysis
#
for x in df_quandl:
    analyze(
        data=data,
        df_quandl=x['data'],
        directory='viz/{x}'.format(x=x['database']),
        directory_report='reports/{x}'.format(x=x['dataset']),
        screen_name=screen_name,
        stopwords=stopwords
    )
