#!/usr/bin/python

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from config import twitter_api as t_creds
from brain.exploratory.sentiment import Sentiment
from consumer.twitter_api.twitter_query import TwitterQuery


def tweet_sn(
    screen_name,
    start_date,
    end_date,
    'full_text'='full_text',
    directory='data/twitter',
    count=600,
    rate_limit=900
):
    '''

    harvest tweets from list of screen names.

    '''

    if not os.path.exists(directory):
        os.makedirs(directory)

    # instantiate api
    t = TwitterQuery(
        t_creds['CONSUMER_KEY'],
        t_creds['CONSUMER_SECRET']
    )

    data = {}
    for i,sn in enumerate(screen_name):
        if Path('{d}/{sn}.csv'.format(d=directory, sn=sn)).is_file():
            data[sn] = pd.read_csv('{d}/{sn}.csv'.format(d=directory, sn=sn))

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
                    count=count,
                    rate_limit=rate_limit
                )

                # sentiment analysis
                s = Sentiment(data[sn], 'full_text')
                data[sn] = pd.concat([s.vader_analysis(), data[sn]], axis=1)
                data[sn].replace('\s+', ' ', regex=True, inplace=True)

                # store locally
                data[sn].to_csv('{d}/{sn}.csv'.format(d='data/twitter', sn=sn))

            except Exception as e:
                print('Error: did not finish \'{sn}\'.'.format(sn=sn))
                print(e)

        # convert to string
        data[sn]['created_at'] = data[sn]['created_at'].astype(str)

        # largest time span
        start = data[screen_name[i]]['created_at'].iloc[0]
        temp_start = datetime.strptime(start.split()[0], '%Y-%m-%d')

        if temp_start < start_date:
            start_date = temp_start

        end = data[screen_name[i]]['created_at'].iloc[-1]
        temp_end = datetime.strptime(end.split()[0], '%Y-%m-%d')
        if temp_end > end_date:
            end_date = temp_end

    return(data, start_date, end_date)
