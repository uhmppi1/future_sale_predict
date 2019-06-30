import keras
import json
import numpy as np
from keras import models
from keras.models import model_from_json
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from model.repeat_layer import RepeatEmbedding

with open('checkpoint/model_1_20190628_1507.json','r') as f:
    model_json = json.load(f)
    # print(model_json)

    model = model_from_json(str(model_json), custom_objects={'RepeatEmbedding':RepeatEmbedding})
    model.load_weights('checkpoint/model_1_20190628_1507.h5')
    model.summary() # 기억을 되살리기 위해서 모델 구조를 출력합니다