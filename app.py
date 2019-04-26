#!/usr/bin/python

#
# this project requires the following packages:
#
#   pip install Twython
#

import re
from consumer.twitter_stream import TwitterStream
from consumer.twitter_query import TwitterQuery
from config import twitter_api as creds
from model.text_classifier import Model as model

# local variables
stream = False

#
# single query
#
q = TwitterQuery(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])

#
# many filters can be implemented with the twitter api:
#
# - https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets
#
# examples:
#
# @q, utf-8, url-encoded search query of 500 characters maximum, including operators.
#     Queries may additionally be limited by complexity.
# @geocode, returns tweets by users located within a given radius of the given latitude
#     and longitude. The location is preferentially taking from the Geotagging API, but
#     will fall back to their Twitter profile.
# @lang, restricts tweets to the given language.
# @result_type, type of search results (i.e. mixed, recent, popular)
#
q.query({
    'q': 'python',
    'result_type': 'popular',
    'count': 10,
    'lang': 'en',
})

#
# single query: timeline of screen name.
#
df_elon = q.query_user('elonmusk')
df_bezos = q.query_user('JeffBezos')
df_overall = df_elon.append(df_bezos)
df_overall.replace({'screen_name': {'elonmusk': 0, 'JeffBezos': 1}})

# reduce to ascii
df_overall['text'] = [re.sub(r'[^\x00-\x7f]', r' ', s) for s in df_overall['text']]

# unigram: perform unigram analysis.
unigram = model(df_overall, key_text='text', key_class='screen_name')

# unigram vectorize
unigram_params = unigram.get_split()
unigram_vectorized = unigram.get_tfidf()

# unigram classifier
model_unigram = unigram.model(
    unigram_vectorized,
    unigram_params['y_train'],
    validate=(unigram_params['X_test'], unigram_params['y_test'])
)

# plot unigram
unigram.plot_cm(filename='cm_unigram.png')

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
