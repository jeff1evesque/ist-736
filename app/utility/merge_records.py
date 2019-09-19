#!/usr/bin/python

import pandas as pd


def merge_records(df, col_1, col_2, drop_empty_col_2=True):
    '''

    merge records of a provided dfframe when a specified condition for
    'col_1' and 'col_2' is satisfied:

    @col_1, current row in this column is null, and previous row notnull.
    @col_2, rows in this column conditionally get appended the previous
        row value given condition for 'col_1'.

    '''

    drop_indices = []
    col_names = df.columns.tolist()

    for i,(idx,row) in enumerate(df.iterrows()):
        if (
            i == 0 and
            col_1 in df and
            pd.isnull(df[col_1].values[i])
        ):
            drop_indices.append(i)

        elif (
            i > 0 and
            col_1 in df and
            col_2 in df and
            pd.isnull(df[col_1].values[i]) and
            pd.notnull(df[col_1].values[i-1]) and
            pd.notnull(df[col_2].values[i])
        ):
            for x in col_names:
                col_idx = df.columns.get_loc(x)
                if x == col_2:
                    df.iat[i, col_idx] = '{a} {b}'.format(
                        a=df.iat[i-1, col_idx],
                        b=df.iat[i, col_idx],
                    )

                else:
                    df.iat[i, col_idx] = df.iat[i-1, col_idx]

            drop_indices.append(i-1)

    if drop_empty_col_2:
        drop_indices.extend(df[df[col_2] == ''].index)

    #
    # drop rows: rows with no tickers and empty col_2.
    #
    if len(drop_indices) > 0:
        target_indices = [df.iloc[[i]].index.values[0]
            for i in drop_indices
                if isinstance(i, int) and i < len(df.index)]

        for x in target_indices:
            if x in df:
                df.drop(x, inplace=True)

    return(df)
