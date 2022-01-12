"""
CTR prediction 에 사용되는 각종 함수들의 모임~~

"""
# Author: Haneul Kim <haneulkim214@gmail.com>
# License: Enliple


import joblib
import os
import pickle

def save_pickle(obj, name, *path):
    """
    Util function for saving object/file as pickle.

    Parameters
    ----------
    obj : Object
        What you want to save as pickle

    name : str
        name of your pickle file

    path : str
    """
    full_path = os.path.join(*path, name)
    joblib.dump(obj, f'{full_path}.pkl')
    print(f"{name} successfully saved!")