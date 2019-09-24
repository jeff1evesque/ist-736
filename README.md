# ist-736

This is a final project for a short 10 week course in text mining. Coding, [visualizations](https://github.com/jeff1evesque/ist-736/tree/master/viz), and overall report were improved after completion of the course, since the project was submitted as one of among several projects within a portfolio requirement for graduation. In general, the project attempts to address several items, including the larger question -- Can Market Sentiment Predict the Stock Market?

## Analysis

To address this overall question, different techniques were applied.

- exploratory analysis: [topic modeling](https://github.com/jeff1evesque/ist-736/blob/master/brain/algorithm/topic_model.py) determines which stock to study
- [sentiment analysis](https://github.com/jeff1evesque/ist-736/blob/master/brain/exploratory/sentiment.py): text corpus are normalized into sentiment scores
- [granger analysis](https://github.com/jeff1evesque/ist-736/blob/master/brain/algorithm/granger.py): find significant sentiment scores and stock index
- timeseries analysis: determine [LSTM](https://github.com/jeff1evesque/ist-736/blob/master/brain/algorithm/lstm.py) and [ARIMA](https://github.com/jeff1evesque/ist-736/blob/master/brain/algorithm/arima.py) comparison for sentiment and stock series

While the main focus of the study were between timeseries models, classification analysis was also performed. Specifically, signal analysis was used as the basis for classification:

- [signal analysis](https://github.com/jeff1evesque/ist-736/blob/master/brain/algorithm/peak_detection.py): apply signal analysis to determine exceeding index points
- [classification analysis](https://github.com/jeff1evesque/ist-736/blob/master/brain/algorithm/text_classifier.py): TF-IDF text corpus (X) trained against signal results

In general, points exceeding the upper limit threshold was binned a value `1`, while points below the lower threshold was binned a value `-1`. This approach provided the target vector (y) when using the TF-IDF corpus (X) during classifcation:

![threshold_animation](https://user-images.githubusercontent.com/2907085/65475387-66335900-de4d-11e9-992e-3d658d11c3f4.gif)

While the exact details of the project can be reviewed from the associated [`write-up.docx`](https://github.com/jeff1evesque/ist-736/blob/master/write-up.docx), the remaining segments in this document will remain succinct.

## Dependencies

This project requires the following packages:

```bash
$ sudo pip install nltk \
    matplotlib \
    twython \
    quandl \
    sklearn \
    scikit-plot \
    statsmodels \
    seaborn \
    wordcloud \
    keras \
    numpy
```

## Data

Two different datasets were acquired via the [Twython](https://twython.readthedocs.io/en/latest/) and [Quandl](https://docs.quandl.com/) API:

- [financial analyst](https://github.com/jeff1evesque/ist-736/tree/master/data/twitter) tweets
- [stock market](https://github.com/jeff1evesque/ist-736/tree/master/data/quandl) index/volume measures

Due to limitations of the twitter API, roughly 3200 tweets could be collected for a given [user timeline](https://developer.twitter.com/en/docs/tweets/timelines/api-reference/get-statuses-user_timeline). However, the quandl data has a much larger limit. This imposed a [limitation](https://github.com/jeff1evesque/ist-736/blob/master/app/join_data.py) upon joining the data. Specifically, only a subset of the twitter corpus was utilized during the analysis.

## Execution

Original aspiration was to complete the codebase using the [app-factory](https://flask.palletsprojects.com/en/1.1.x/patterns/appfactories/). Due to time constraint, the codebase was not expanded as an application. However, a provided [`config-TEMPLATE.py`](https://github.com/jeff1evesque/ist-736/blob/master/config-TEMPLATE.py) is required a minimum to be copied as `config.py` in the same directory. If additional twitter user timelines, or quandl stock index would be studied, the contents of the configuration files need to match API tokens for each of the service providers. However, to run the codebase to reflect the choices made in this study, then no API keys are needed. Instead, the contents of the [`app.py`](https://github.com/jeff1evesque/ist-736/blob/master/app.py) need to be properly commented out. Specifically, only one analysis can be performed at a given time. Moreover, timeseries sentiment models (consisting of both ARIMA and LSTM) has an added constraint. Specifically, only one stock code can be implemented at a given time:

```python
screen_name = [
    'jimcramer',
    'ReformedBroker',
    'TheStalwart',
    'LizAnnSonders',
    'SJosephBurns'
]
codes = [
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
##    ('EIA', 'PET_RWTC_D'),
##    ('WFC', 'PR_CON_15YFIXED_IR'),
##    ('WFC', 'PR_CON_30YFIXED_APR')
]
```

This is largely due to an exponentiating [memory requirement](https://github.com/jeff1evesque/ist-736/issues/125), due to keeping multiple trained neural networks in memory. Should this codebase be extended to an application, the latter issue would need to be resolved. Nevertheless, additional configurations can be adjusted in the same file, including the number of epochs, lstm cells, signal analysis threshold (i.e. `classify_threshold`), and TF-IDF feature reduction for classification (i.e. `classify_chi2`) can be made. After dependenices and necessary changes have been made, the script can be executed in a stepwise fashion:

```bash
$ pwd
/path/to/web-projects/ist-736
$ python app.py
```
