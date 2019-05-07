#!/usr/bin/python

#
# this project requires the following packages:
#
#   pip install Twython
#

import os
import re
from pathlib import Path
import pandas as pd
from config import twitter_api as creds
from consumer.twitter_query import TwitterQuery
from view.exploratory import explore
from exploratory.sentiment import Sentiment

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
stopwords=['http', 'https', 'nhttps', 'RT', 'amp', 'co', 'TheStreet']
stopwords.extend(screen_name)

#
# create directories
#
if not os.path.exists('data/twitter'):
    os.makedirs('data/twitter')

#
# instantiate api
#
q = TwitterQuery(
    creds['CONSUMER_KEY'],
    creds['CONSUMER_SECRET']
)

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
            data[sn] = q.query_user(
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
                rate_limit=1
            )

            # sentiment analysis
            s = Sentiment(data[sn], 'full_text')
            data[sn] = s.vader_analysis()

            # store locally
            data[sn].to_csv('data/twitter/{sn}.csv'.format(sn=sn))

        except Exception as e:
            print('Error: did not finish \'{sn}\'.'.format(sn=sn))
            print(e)
#
# preprocess: combine and clean dataframe(s)
#
df = pd.concat(data)
df.replace({'screen_name': {v:i for i,v in enumerate([*screen_name])}})

#
# exploratory
#
explore(df, stopwords=stopwords, sent_cases={'screen_name': screen_name})
