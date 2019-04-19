#!/usr/bin/python

from consumer.twitter import Twitter


# instantiate stream
stream = Twitter()
stream.set_query()
stream.statuses.filter(track='python')
