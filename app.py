#!/usr/bin/python

import json
from consumer.twitter_stream import TwitterStream
from consumer.twitter_query import TwitterQuery
from config import twitter_api as creds

#
# this project requires the following packages:
#
#   - Twython
#

# local variables
stream = False

#
# single query
#
q = TwitterQuery(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
q.query({
    'q': 'learn python',
    'result_type': 'popular',
    'count': 10,
    'lang': 'en',
})

#
# stream query
#
if stream:
    s = TwitterStream(
        creds['CONSUMER_KEY'],
        creds['CONSUMER_SECRET'],
        creds['ACCESS_TOKEN'],
        creds['ACCESS_SECRET']
    )

    #
    # many filters can be implemented with the twitter api:
    #
    # - https://developer.twitter.com/en/docs/tweets/filter-realtime/guides/basic-stream-parameters.html
    #
    # examples:
    #
    # @follow, comma-separated list of user IDs, indicating the users whose Tweets
    #     should be delivered on the stream.
    # @track, comma-separated list of phrases which will be used to determine what
    #     Tweets will be delivered on the stream.
    # @locations, comma-separated list of longitude,latitude pairs specifying a set
    #     of bounding boxes to filter Tweets by.
    #
    s.statuses.filter(track='python')
