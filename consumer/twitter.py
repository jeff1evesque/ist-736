#!/usr/bin/python

import csv
import json
from twython import Twython  
from twython import TwythonStreamer


class Twitter(TwythonStreamer):
    '''

    wrapper to the twython package.

    '''

    def __init__(
        self,
        config='{}/config.py'.format(Path(__file__).resolve().parents[1])
    ):
        '''

        define class variables.

        '''

        # inherit base class
        super(Twitter, self).__init__()

        # twitter configurations
        with open(config, 'r') as fp:
            creds = json.load(fp)

        self.python_tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])

    def get_dict_val(self, d, keys):
        '''

        return value of nested dict using provided list of keys.

        '''

        for k in keys:
            d = d.get(k, None)
            if d is None:
                return(None)
        return(d)

    def get_dict_path(self, d):
        '''

        given a dict:

            d = {
                'A': {
                    'leaf-1': 'value-1',
                    'node-1': {'leaf-sub-1': 'sub-value-1'}
                }
            }

        a nested list is generated and returned:

            [
                ['A', 'leaf-1', 'value-1'],
                ['A', 'node-1', 'leaf-sub-1', 'sub-value-1']
            ]

        '''

        temp = []
        result = []
        for k,v in d.items():
            if isinstance(v, dict):
                temp.append(k)
                get_dict_path(v)
            else:
                if isinstance(v, list):
                    [result.append(x) for x in v]
                    temp = []
                else:
                    temp.append(v)
                    result.append(temp)
                    temp = []

        return(result)

    def set_query(
        self,
        params=[{'user': ['screen_name']}, 'created_at', 'text']
        sorted=None
    ):
        '''

        search tweets using provided parameters and default credentials.

        @keys, list of lists, recursive params key through end value.
        @sorted, dict of panda 'sort_values'.

        Note: additional search arguments, as well as full response
              'statuses' can be utilized and referenced:

            - https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets.html

        '''

        keys = []
        [keys.extend(get_dict_path(k)) if isinstance(k, dict) else [k] for k in params]

        self.params = {x[-1]: [] for x in keys}
        [self.params[k].append(get_dict_val(keys, i)) for i, (k,v) in enumerate(self.params.items())]

    def on_success(self, data):
        '''

        required 'TwythonStreamer' method called when twitter returns data. 

        '''

        tweet_data = self.result
        self.save_to_csv(tweet_data)

    def on_error(self, status_code, data):
        '''

        required 'TwythonStreamer' method called when twitter returns an error. 

        '''

        print(status_code, data)
        self.disconnect()

    def save_to_csv(self, tweet):
        '''

        optional 'TwythonStreamer' method to store tweets into a file. 

        '''

        with open(r'saved_tweets.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(list(tweet.values()))    
