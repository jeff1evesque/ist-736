#!/usr/bin/python

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from config import quandl_api as q_creds
from consumer.quandl_api.quandl_query import QuandlQuery


def quandl(
    codes=[('NASDAQOMX', 'COMP-NASDAQ')],
    start_date=None,
    end_date=None,
    collapse='daily',
    directory='data/quandl'
):
    '''

    harvest quandl data.

    '''

    if not os.path.exists(directory):
        os.makedirs(directory)

    # instantiate api
    q = QuandlQuery(q_creds['API_KEY'])

    #
    # harvest quandl
    #
    data = []
    for x in codes:
        if Path('{d}/{f}'.format(d=directory, f=x[1])).is_file():
            df = pd.read_csv('{d}/{f}'.format(d=directory, f=x[1]))

        else:
            df = q.get_ts(
                database_code=x[0],
                dataset_code=x[1],
                start_date=start_date,
                end_date=end_date,
                collapse=collapse
            )
            df.to_csv('{d}/{f}.csv'.format(d=directory, f=x[1]))

        data.append(df)

    return(data)
