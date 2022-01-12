"""
전처리, 학습된 모델을 사용하여 API 에 serving 하는 script입니다.

"""
# Author: Haneul Kim <haneulkim214@gmail.com>
# License: Enliple

from datetime import datetime
from flask import Flask, request, render_template, jsonify
from flask_restful import Api
import joblib
import numpy as np
import os
import pandas as pd

from api.util import check_json
from const import tsfmr_path, model_path, model2use, tsfmr2use, datetime_col_name
from error_handler import Errors, ModelError, InputError
from preprocessing.date_pp import split_datetime


print("Time to serve API~")
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


# 학습때 저장된것들 불러오기.
tsfmr_full_path = os.path.join(tsfmr_path, f"{tsfmr2use}.pkl")
tsfmr = joblib.load(tsfmr_full_path)

model_full_path = os.path.join(model_path, f"{model2use}.pkl")
binary_clf = joblib.load(model_full_path)

@app.route("/click_prediction", methods=["POST"])
def click_prediction():
    inp_data = request.get_json()
    check_json(inp_data)

    # ??? json 자체에서 예측하는방법을 고려해야함 for efficiency
    one_row_df = pd.DataFrame({key: [item] for key, item in inp_data.items()})
    one_row_df = split_datetime(one_row_df, datetime_col_name)
    one_row_df = tsfmr.transform(one_row_df)
    click_pred = binary_clf.predict(one_row_df)[0]

    try:
        click_pred_p = binary_clf.predict_proba(one_row_df)[0][-1]
        click_pred_p = round(click_pred_p*100, 2)
    except AttributeError:
        click_pred_p = ""

    return jsonify({"expected_click":click_pred, "click_%":click_pred_p,
                    "model":model2use, "status":"00", "message":"success"})

@app.errorhandler(Errors)
def error_handler(error):
    return error.response()


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=5000)