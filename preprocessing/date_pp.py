"""
날짜, 시간 관련된 전처리

"""
# Author: Haneul Kim <haneulkim214@gmail.com>
# License: Enliple


import numpy as np
import pandas as pd


def is_weekday(x):
    """1 = weekday"""
    if x <= 4:
        return 1
    else:
        return 0

def split_datetime(df, dt_col_nm):
    """
    1. %Y-%m-%d %H:%M:%S 을 각 column 으로 나누어준다.
    2. 평일/주말(1/0) column 추가
    2. datetime column 삭제

    ex: 2021-11-18 18:25:55 -> 2021, 11, 18, 18, 25 column 들로.

    Parameters
    ----------
    dt_col_nm : str
             datetime column name.
    """
    df[dt_col_nm] = pd.to_datetime(df[dt_col_nm])
    df["year"] = df[dt_col_nm].dt.year
    df["month"] = df[dt_col_nm].dt.month.astype(np.int8)
    df["day"] = df[dt_col_nm].dt.day.astype(np.int8)
    df["weekday"] = df[dt_col_nm].dt.dayofweek.apply(is_weekday).astype(np.int8)

    return df.drop(columns=[dt_col_nm])