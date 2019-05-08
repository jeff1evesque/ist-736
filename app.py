#!/usr/bin/python

#
# this project requires the following packages:
#
#   pip install Twython
#   pip install Quandl
#

import os
import re
from pathlib import Path
import pandas as pd
from config import twitter_api as t_creds
from config import quandl_api as q_creds
from consumer.twitter_query import TwitterQuery
from consumer.quandl_query import QuandlQuery
from view.exploratory import explore
from exploratory.sentiment import Sentiment
from datetime import datetime

#
# local variables
#
data = {}
screen_name = [
    'jimcramer',
    'ReformedBroker',
    'TheStalwart',
    'LizAnnSonders',
    'SJosephBurns'
]
stopwords=[
    'http',
    'https',
    'nhttps',
    'RT',
    'amp',
    'co',
    'TheStreet'
]
stopwords.extend(screen_name)

#
# create directories
#
if not os.path.exists('data/twitter'):
    os.makedirs('data/twitter')

if not os.path.exists('data/quandl'):
    os.makedirs('data/quandl')

#
# instantiate api
#
t = TwitterQuery(
    t_creds['CONSUMER_KEY'],
    t_creds['CONSUMER_SECRET']
)
q = QuandlQuery(q_creds['API_KEY'])

#
# combine quandl with tweets
#
start_date = datetime(3000, 12, 25)
end_date = datetime(1000, 12, 25)

for i,sn in enumerate(screen_name):
    #
    # create directories
    #
    if not os.path.exists('viz/{sn}'.format(sn=sn)):
        os.makedirs('viz/{sn}'.format(sn=sn))

    #
    # harvest tweets
    #
    if Path('data/twitter/{sn}.csv'.format(sn=sn)).is_file():
        data[sn] = pd.read_csv('data/twitter/{sn}.csv'.format(sn=sn))

    else:
        try:
            data[sn] = t.query_user(
                sn,
                params=[
                    {'user': ['screen_name']},
                    'created_at',
                    'full_text',
                    {'retweeted_status': ['full_text']},
                    'retweet_count',
                    'favorite_count',
                    {'entities': ['user_mentions']}
                ],
                count=600,
                rate_limit=900
            )

            # sentiment analysis
            s = Sentiment(data[sn], 'full_text')
            data[sn] = pd.concat([s.vader_analysis(), data[sn]], axis=1)

            # store locally
            data[sn].to_csv('data/twitter/{sn}.csv'.format(sn=sn))

        except Exception as e:
            print('Error: did not finish \'{sn}\'.'.format(sn=sn))
            print(e)

    # largest time span
    start = data[screen_name[i]]['created_at'].iloc[0]
    temp_start = datetime.strptime(start.split()[0], '%Y-%m-%d')
    if temp_start < start_date:
        start_date = temp_start

    end = data[screen_name[i]]['created_at'].iloc[-1]
    temp_end = datetime.strptime(end.split()[0], '%Y-%m-%d')
    if temp_end > end_date:
        end_date = temp_end

#
# harvest quandl: using the maximum twitter date range
#
if Path('data/quandl/nasdaq.csv').is_file():
    df_nasdaq = pd.read_csv('data/quandl/nasdaq.csv')

else:
    df_nasdaq = q.get_ts(start_date=start_date, end_date=end_date)
    df_nasdaq.to_csv('data/quandl/nasdaq.csv')

#
# preprocess: combine and clean dataframe(s)
#
df = pd.concat(data)
df.replace({'screen_name': {v:i for i,v in enumerate([*screen_name])}})

#
# classification: left join on twitter dataset(s).
#
for i,sn in enumerate(screen_name):
    # merge with consistent date format
    date[sn]['created_at'] = datetime.strftime(
        '%Y-%m-%d',
        datetime.strptime(df['date'], '%Y-%m-%d')
    )

    merged_data = pd.merge(
        data[sn],
        df_nasdaq,
        left_on='created_at',
        right_on='Trade Date',
        how='left'
    )

#
# exploratory
#
explore(df, stopwords=stopwords, sent_cases={'screen_name': screen_name})
