"""
해당 프로젝트에 사용 되는 constant 들을 저장하기 위함.

"""
# Author: Haneul Kim <haneulkim214@gmail.com>
# License: Enliple


# Model data
file_dir = "data/raw_data"
file_csv = "CW_MEDIA_CLICKVIEW_LOG_CTR_데이터_추출1.csv"

# Model list: logistic, xgboost_clf, decision_tree_clf, mlp_clf
# Transformer list: linear_tsfmr, non_linear_tsfmr

model2use = "xgboost_clf"
tsfmr2use = "non_linear_tsfmr"

# model2use = "logistic"
# tsfmr2use = "linear_tsfmr"

col_order = ["createdDate", "mediaId", "inventoryId", "adverId", "adType", "frameId", "auid", "logType", # ordered
             # not ordered for now
             "adProduct", "adCampain", "productCode", "cpoint", "mpoint",
             'remoteIp', 'platform', 'device', 'browser', 'freqLog',
             'tTime', 'kno', 'kwrdSeq', 'gender', 'age', 'osCode', 'price',
             'frameCombiKey'
            ]

## Column names
# Forward-Selection 방법론으로 *파악된* feature 하나하나 늘려가며 사용한다.
cate_ord_cols = ["mediaId", "inventoryId", "adverId"]
cate_ohe_cols = ["platform"]
num_cols = ["cpoint", "mpoint", "freqLog", "tTime"]
other_cols = ["year", "month", "day", "weekday"]
target_col_nm = "click"
use_cols = cate_ord_cols + cate_ohe_cols + num_cols + other_cols + [target_col_nm]
datetime_col_name = "createdDate"



# 학습된 각종 Object 들 저장 path. ex:학습된 Scaler.
tsfmr_path = "data/trained_data"
model_path = "data/trained_data"

# Imbalanced_label 전처리 관련 const
replace = True
p = -1


