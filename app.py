#!/usr/bin/python

from consumer.twitter_stream import TwitterStream
from consumer.twitter_query import TwitterQuery


# local variables
stream = False

# twitter configurations
with open('config.py', 'r') as fp:
    creds = json.load(fp)['twitter_api']

#
# single query
#
q = TwitterQuery(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
q.query()

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
