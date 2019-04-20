#!/usr/bin/python

from twython import Twython
import pandas as pd


class TwitterQuery():     
    '''

    requires instance of TwitterStream.

    '''

    def __init__(self, key, secret):
        '''

        define class variables.

        '''

        self.conn = Twython(key, secret)

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
                self.get_dict_path(v)
            else:
                if isinstance(v, list):
                    [result.append(x) for x in v]
                    temp = []
                else:
                    temp.append(v)
                    result.append(temp)
                    temp = []

        return(result)

    def query(
        self,
        query,
        params=[{'user': ['screen_name']}, 'created_at', 'text'],
        sorted=None
    ):
        '''

        search tweets using provided parameters and default credentials.

        @query, query parameters of the form:

            {
                'q': 'learn python',  
                'result_type': 'popular',
                'count': 10,
                'lang': 'en',
            }

        @keys, list of lists, recursive params key through end value.
        @sorted, dict of panda 'sort_values'.

        Note: additional search arguments, as well as full response
              'statuses' can be utilized and referenced:

            - https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets.html

        '''

        keys = []
        [keys.extend(self.get_dict_path(k)) if isinstance(k, dict) else [k] for k in params]
        self.result = {x[-1]: [] for x in keys}

        # query
        for status in self.conn.search(**query)['statuses']:
            [self.result[k].append(get_dict_val(status, keys[i])) for i, (k,v) in enumerate(self.result.items())]

        self.df = pd.DataFrame(self.result)
        return(self.df)
