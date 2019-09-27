#!/usr/bin/python

'''

TEMPLATE: copy this file as 'config.py', then change any of the below
          key-value configuration as needed.

'''

#
# api controls
#
twitter_api = {
    'CONSUMER_KEY': 'REPLACE-ME',
    'CONSUMER_SECRET': 'REPLACE-ME',
    'ACCESS_TOKEN': 'REPLACE-ME',
    'ACCESS_SECRET': 'REPLACE-ME'
}

quandl_api = {
    'API_KEY': 'REPLACE-ME'
}

#
# model controls
#
model_control = {
    'analysis_explore': False,
    'analysis_granger': False,
    'analysis_ts_stock': True,
    'analysis_ts_sentiment': False,
    'analysis_classify': False
}

model_config = {
    'classify_index': 'full_text',
    'ts_index': 'value',
    'arima_auto_scale': None,
    'lstm_epochs': 750,
    'lstm_num_cells': 4,
    'lstm_units': 50,
    'lstm_dropout': 0.2,
    'classify_threshold': [0.5],
    'classify_chi2': 100
}

save_result = {
    'save_lstm': True,
    'save_lstm_log': True,
    'save_lstm_log_plot': True,
    'save_model_plots': True
}

#
# application controls
#
twitter_accounts = [
    'jimcramer',
    'ReformedBroker',
    'TheStalwart',
    'LizAnnSonders',
    'SJosephBurns'
]

stock_codes = [
    ('BATS', 'BATS_AAPL'),
##    ('BATS', 'BATS_AMZN'),
##    ('BATS', 'BATS_GOOGL'),
##    ('BATS', 'BATS_MMT'),
##    ('BATS', 'BATS_NFLX'),
##    ('CHRIS', 'CBOE_VX1'),
##    ('NASDAQOMX', 'COMP-NASDAQ'),
##    ('FINRA', 'FNYX_MMM'),
##    ('FINRA', 'FNSQ_SPY'),
##    ('FINRA', 'FNYX_QQQ'),
##      ('EIA', 'PET_RWTC_D'),
##    ('WFC', 'PR_CON_15YFIXED_IR'),
##    ('WFC', 'PR_CON_30YFIXED_APR')
]

sentiments = ['negative', 'neutral', 'positive']

drop_cols = [
    'compound',
    'retweet_count',
    'favorite_count',
    'user_mentions',
    'Short Volume'
]
