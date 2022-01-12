"""
Classification 관련 모델들


"""
# Author: Haneul Kim <haneulkim214@gmail.com>
# License: Enliple

import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from xgboost import XGBClassifier

logistic = LogisticRegression(penalty="l2", max_iter=1000)
xgbm_clf = XGBClassifier()
dt_clf = DecisionTreeClassifier(min_samples_split=10000)
mlp_clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32),
                       activation='relu')

# Neural Network
# nn_clf = keras.Sequential([
#     keras.layers.Dense(13, input_shape=(13, ), activation='relu'), # input layer
#     keras.layers.Dense(1, activation='sigmoid'), # output layer
# ])
# nn_clf.compile(optimizer='adam',
#                loss="binary_crossentropy")


class BinaryClf:
    """
    Models for Binary Classification
    """
    def __init__(self, logistic, xgbm_clf, dt_clf, nn_clf):
        """
        Initializng models
        """
        self.logistic = logistic
        self.xgbm = xgbm_clf
        self.dt = dt_clf
        self.neural_net = nn_clf
    pass
