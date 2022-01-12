"""
CTR 예측을 하기위해 필요한 전처리, 모델 학습을 진행 하는
main.py 입니다.


"""
# Author: Haneul Kim <haneulkim214@gmail.com>
# License: Enliple

import os
import time
from datetime import datetime, timedelta


if __name__ == '__main__':
    os.system("python3 train.py")
    os.system("python3 flask_api.py")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
