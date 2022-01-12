"""
Encoder, Scaler, 등등 데이터 전처리에 사용되는 각종
transformer 들을 관리하기 위함

"""
# Author: Haneul Kim <haneulkim214@gmail.com>
# License: Enliple

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

from const import cate_ord_cols, cate_ohe_cols, num_cols


linear_tsfmr = make_column_transformer(
                    (OneHotEncoder(), cate_ohe_cols),
                    (OrdinalEncoder(), cate_ord_cols),
                    (StandardScaler(), num_cols),
                    remainder='passthrough')

non_linear_tsfmr = make_column_transformer(
                    (OneHotEncoder(), cate_ohe_cols),
                    (OrdinalEncoder(), cate_ord_cols),
                    (StandardScaler(), num_cols),
                    remainder='passthrough')