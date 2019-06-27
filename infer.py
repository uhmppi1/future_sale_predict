import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


def infer_testset(model, X_test, y_test):
    y_hats = []
    test_size = X_test[0].shape[0]   # X1_test의 길이
    for i in range(test_size):
        x_input = [np.expand_dims(X_test[j][i], axis=0) for j in range(len(X_test))]
        cur_y_hat = model.predict(x_input, verbose=0)
        cur_y_hat = np.squeeze(cur_y_hat)
        print('TEST DATA %d : predicted=%f, ground_truth=%f' % (i, cur_y_hat, y_test[i]))
        y_hats.append(cur_y_hat)

    return y_hats