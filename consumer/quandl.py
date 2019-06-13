#!/usr/bin/python

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from config import quandl_api as q_creds
from consumer.quandl_api.quandl_query import QuandlQuery


def quandl(
    database_code='NASDAQOMX',
    dataset_code='COMP-NASDAQ',
    start_date=None,
    end_date=None,
    collapse='daily',
    filepath='data/quandl/nasdaq.csv'):
    '''

    harvest quandl data.

    '''

    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # instantiate api
    q = QuandlQuery(q_creds['API_KEY'])

    # default: maximum twitter date range
    if not start_date:
        start_date = datetime(3000, 12, 25)
    if not end_date:
        end_date = datetime(1000, 12, 25)

    #
    # harvest quandl
    #
    if Path(filepath).is_file():
        df = pd.read_csv(filepath)

    else:
        df = q.get_ts(start_date=start_date, end_date=end_date)
        df.to_csv(filepath)

    return(df)
