#!/usr/bin/python

import os
import pandas as pd
from datetime import datetime
from brain.view.exploratory import explore as expl
from brain.controller.topic_model import topic_model


def explore(
    data,
    screen_name,
    target='created_at',
    stopwords=[],
    stopwords_topics=[],
    directory='viz',
    explore_sn=True,
    explore_topic=True,
    explore_overall=True
):
    '''

    general exploration on provided screen names.

    '''

    for i,sn in enumerate(screen_name):
        # convert to string
        data[sn][target] = data[sn][target].astype(str)

        #
        # exploratory: wordcloud + sentiment
        #
        if explore_sn:
            expl(
                data[sn],
                stopwords=stopwords,
                directory='{d}/{sn}'.format(d=directory, sn=sn)
            )

        #
        # topic modeling
        #
        if explore_topic:
            topic_model(
                data[sn],
                rotation=0,
                stopwords=list(set(stopwords_topics)),
                num_topics=10,
                random_state=1,
                flag_nmf=False,
                directory='{d}/{sn}'.format(d=directory, sn=sn)
            )

    #
    # exploratory (overall): wordcloud + sentiment
    #
    df = pd.concat(data).reset_index()

    if explore_overall:
        expl(
            df,
            stopwords=stopwords_topics,
            sent_cases={'screen_name': screen_name}
        )
