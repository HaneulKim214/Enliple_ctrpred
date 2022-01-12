"""
Imbalanced Dataset 처리하는 방법론들

"""
# Author: Haneul Kim <haneulkim214@gmail.com>
# License: Enliple

import numpy as np
import pandas as pd


class BinaryImbClsRS:
    """
    Handling Imbalanced data in *Train_set*

    Parameters
    ----------
    df : DataFrame
        Dataset for training

    target_col : str
               target column name

    pos_samples : DataFrame
               DataFrame containing rows with label = 1

    neg_samples : DataFrame
               DataFrame containing rows with label = 0

    rep : Bool
      replace data when randomly sampling if True
      else False
    """
    def __init__(self, df, target_col, rep):
        self.df = df
        self.target_col = target_col
        self.rep = rep
        self.pos_samples = self.df.loc[self.df[target_col] == 1].copy()
        self.neg_samples = self.df.loc[self.df[target_col] == 0].copy()

    def random_oversample(self, p):
        """
        if p > 0:
        Randomly oversample p% amount of majority class for minority class,
        so oversample_df will contain neg_sample + alpha

        if p = -1:
        oversample minority class to same size as majority class

        ex: class0 = 100, class1=10, then if p=0.5 oversample upto class1=50.

        Parameters
        ----------
        p : float
          len(neg_samples)*p = # of pos_samples.
        """
        if p == -1:
            oversample_df = self.pos_samples.sample(
                len(self.neg_samples) - len(self.pos_samples), replace=self.rep
            )
        else:
            oversample_df = self.pos_samples.sample(
                np.ceil(len(self.neg_samples)*p), replace=self.rep
            )
        return pd.concat([self.df, oversample_df])

    def random_undersample(self, p):
        """
        if p > 0:
        Only use p% of majority class,
        so undersample_df will contain subset of majority class

        if p = -1:
        undersample majority class to same size as minority class

        Parameters
        ----------
        p : float
          len(pos_samples)*p = # of neg_samples
        """
        if p == -1:
            undersample_df = self.neg_samples.sample(
                len(self.pos_samples), replace=self.rep
            )
        else:
            undersample_df = self.neg_samples.sample(
                np.ceil(len(self.pos_samples)*p), replace=self.rep
            )
        # ??? concat 하면 각 label 마다 숫자가 틀리다 => NaN 이 생성되어있다
        return pd.concat([self.pos_samples, undersample_df])


class DfPreprocess:
    """
    dataframe을 사용에 알맞게 전처리 합니다.
    """
    def __init__(self):
        pass

    def get_ymd(self, df, col_name):
        """
        Separate date column into year, month, date column.

        Returns
        -------
        dataframe : df with columns separated.
        """

        df["year"] = df[col_name].dt.year
        df["month"] = df[col_name].dt.month
        df["day"] = df[col_name].dt.day

        df.drop(columns=[col_name], inplace=True)
        return df

    def is_weekday(self):
        """
        Check if date is weekday or weekend(including holidays)

        """
        pass
