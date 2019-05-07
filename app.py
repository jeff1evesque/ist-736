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

#
# local variables
#
data = []
screen_name = [
    'jimcramer',
    'ReformedBroker',
    'TheStalwart',
    'LizAnnSonders',
    'SJosephBurns'
]

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

for sn in screen_name:
    #
    # create directories
    #
    if not os.path.exists('viz/{sn}'.format(sn=sn)):
        os.makedirs('viz/{sn}'.format(sn=sn))

    #
    # harvest tweets
    #
    if Path('data/twitter/{sn}'.format(sn=sn)).is_file():
        data[sn] = pd.read_csv('data/twitter/{sn}'.format(sn=sn))

    else:
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
            rate_limit=900
        )

        data[sn].to_csv('data/twitter/{sn}'.format(sn=sn))

#
# single query: timeline of screen name.
#
df = pd.concat(data)
df.replace({'screen_name': {v:i for i,v in enumerate(screen_name)}})

#
# exploratory
#
explore(df, sent_cases={'screen_name': screen_name})
