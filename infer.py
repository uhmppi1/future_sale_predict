import numpy as np
from numpy.random import seed
import json
import matplotlib.pyplot as plt
from dataset import data_loader
from model import lstm_with_shop_cate_embedding
from model.repeat_layer import RepeatEmbedding
from keras.models import model_from_json
from keras.optimizers import Adam
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

data_pickle_file_path = 'dataset/pickle/dataset_12step_shop_cate.pkl'
model_checkpoint_name = 'model_1_20190701_1032'

seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

def infer_testset(model, X_test, y_test):
    y_hats = []
    test_size = X_test[0].shape[0] # X1_test의 길이
    for i in range(test_size):
        x_input = [np.expand_dims(X_test[j][i], axis=0) for j in range(len(X_test))]
        cur_y_hat = model.predict(x_input, verbose=0)
        cur_y_hat = np.squeeze(cur_y_hat)
        print('TEST DATA %d : predicted=%f, ground_truth=%f' % (i, cur_y_hat, y_test[i]))
        y_hats.append(cur_y_hat)

    return np.array(y_hats)


def print_result(y_hats, y_test, title=''):
    y_hats = np.array(y_hats)
    y_test = np.array(y_test)
    rmse = np.sqrt(sum((y_hats - y_test) ** 2) / len(y_test))
    print('=========================')
    print('%s RMSE: %.4f' % (title, rmse))
    print('=========================')



if __name__ == "__main__":
    x_seq_len = 12

    dataloader = data_loader.DataLoader(data_pickle_file_path)

    (X1_train, X2_train, X3_train, y_train), (X1_val, X2_val, X3_val, y_val) = dataloader.load_data(x_seq_len)

    print(X1_val.shape)
    print(X2_val.shape)
    print(X3_val.shape)
    print(y_val.shape)

    X_val = [X1_val, X2_val, X3_val]

    shop_num = dataloader.num_shop()
    cate_num = dataloader.num_item_category()

    with open('checkpoint/%s.json' % model_checkpoint_name, 'r') as f:
        model_json = json.load(f)

    model = model_from_json(str(model_json), custom_objects={'RepeatEmbedding':RepeatEmbedding})
    model.load_weights('checkpoint/%s.h5' % model_checkpoint_name)
    model.summary()
    model.compile(optimizer=Adam(lr=0.0001, clipnorm=0.01), loss='mse', metrics=['mse'])

    y_hats = infer_testset(model, X_val, y_val)

    y_val_invscaled = dataloader.get_invertscaled_values(y_val)
    y_hats_invscaled = dataloader.get_invertscaled_values(y_hats)

    print_result(y_hats_invscaled, y_val_invscaled)