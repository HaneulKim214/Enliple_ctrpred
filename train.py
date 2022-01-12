"""
# train.py :
## Main : 예측 전 모든 전처리를 하기 위한 Script
## Leader : Haneul Kim <haneulkim214@gmail.com>
## Sub : Jaemin Kim
## Guide : PEP-8
## License: Enliple
## Idea : 
    1. 함수 영역 : preprocessing 전처리 패키지, util 사용자 정의 함수 패키지 디렉토리를 helps/__init__.py, preprocessing.py, util.py 이렇게 한 곳에서 관리하는 것이 어떨지?
    2. 데이터 영역 : data/ 디렉토리 밑으로 raw_data, trained_data 관리하는 것이 어떨지?
    3. 설정 영역 : config.py 와 const.py의 구조를 합치고 부분 클래스화 시켜서 관리하는 것이 어떨지?
"""

# Python Libs
import joblib
import numpy as np
import os
import pandas as pd
import pickle
import pymysql

from util import save_pickle

# User define Libs
from const import model2use, tsfmr2use, col_order, target_col_nm, use_cols, tsfmr_path, model_path, \
    replace, p, datetime_col_name, file_csv, file_dir
from datetime import datetime
from preprocessing.date_pp import split_datetime
from preprocessing.transform import linear_tsfmr, non_linear_tsfmr
from preprocessing.imbalanced_label import BinaryImbClsRS
from models.classification import logistic, xgbm_clf, dt_clf, mlp_clf


print("Training in progress")
# Make parent path
parent_path = os.path.dirname(os.path.realpath(__file__))

# Read CSV
csv_file_path_lc = os.path.join(parent_path, file_dir, file_csv)
df = pd.read_csv(csv_file_path_lc, sep=",", usecols=col_order, na_values=["NONE", "NULL", "nan"])

# Missing Value imputation => df.replace("NONE|NULL|nan", np.nan, regex=True, inplace=True) 차후 정규식으로 상세 필터링
df.dropna(subset=["gender", "age"], axis="index", inplace=True)

# Date related
df = split_datetime(df, datetime_col_name)

# Drop Columns => Drop columns를 생략하고 use_cols로 마스킹만 하는 방향이 어떨까?
df["click"] = df["logType"].replace({"V":0, "C":1})
df.drop(columns=[
    "frameId", # np.NaN 처리 하기전까지만 drop
    "adType", # df CW 만 사용하기때문
    "auid", # V-C Join 할때만 사용.
    "browser",
    "logType",
    "remoteIp",
    "device", #platform 이랑 동일
    "price", # 전부 0
    "kno", # 전부 0
    "kwrdSeq", # 전부 0, np.NaN
    "frameCombiKey", # 전부 np.NaN
], inplace=True)

df = df[use_cols]
y = df[target_col_nm]
X = df.drop(columns=[target_col_nm])

# Scaling, encoding
tsfmr_dict = {"linear_tsfmr":linear_tsfmr, "non_linear_tsfmr":non_linear_tsfmr}
tsfmr = tsfmr_dict[tsfmr2use]
tsfmr.fit(X)
save_pickle(tsfmr, tsfmr2use, parent_path, tsfmr_path)
trns_X = tsfmr.transform(X) # np.array

# Undersampling
trns_X_df = pd.DataFrame(trns_X)
trns_df = pd.concat([trns_X_df, y], axis=1) #??? 여기서 지금 NaN이 발생한다.
trns_df.dropna(inplace=True)
imb_pp = BinaryImbClsRS(trns_df, target_col_nm, replace)
df_us = imb_pp.random_undersample(p)
y = df_us[target_col_nm]
X = df_us.drop(columns=[target_col_nm])

# Model training
model_dict = {"logistic":logistic, "xgboost_clf":xgbm_clf,
              "decision_tree_clf":dt_clf, "mlp_clf":mlp_clf}
binary_clf = model_dict[model2use]
binary_clf.fit(X, y)
save_pickle(binary_clf, model2use, parent_path, model_path)
print("Training finished")






