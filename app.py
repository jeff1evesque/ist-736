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
    stopwords_topics=stopwords_topics,
    start_date=start_date,
    end_date=end_date
)

#
# harvest quandl + analyze
#
for x in ['COMP-NASDAQ', 'QQQ']:
    df_quandl = quandl(dataset_code=x, start_date, end_date)
    analyze(
        data,
        x,
        directory='viz/{x}'.format(x),
        screen_name,
        stopwords
    )
