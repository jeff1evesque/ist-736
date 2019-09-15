#!/usr/bin/python


def drop_columns(df, cols, drop_cols_regex=None):
    '''

    drop specified columns from provided dataframe.

    '''

    if cols:
        cols_exists = [c for c in cols if c in df]
        df.drop(cols_exists, axis=1, inplace=True)

    if drop_cols_regex:
        if isinstance(drop_cols_regex, (list, set, tuple)):
            for x in drop_cols_regex:
                df = df[df.columns.drop(
                    list(df.filter(regex=x))
                )]

        else:
            df = df[df.columns.drop(
                list(df.filter(regex=drop_cols_regex))
            )]

    return(df)
