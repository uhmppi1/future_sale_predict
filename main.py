from dataset import data_loader
from model import lstm_with_shop_cate_embedding
import train, infer

from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

if __name__ == "__main__":
    x_seq_len = 12
    train_ratio = 0.8

    dataloader = data_loader.DataLoader()

    (X1_train, X2_train, X3_train, y_train), (X1_val, X2_val, X3_val, y_val) = data_loader.load_data(x_seq_len, train_ratio)

    print(X1_train.shape)
    print(X2_train.shape)
    print(X3_train.shape)
    print(y_train.shape)
    print(X1_val.shape)
    print(X2_val.shape)
    print(X3_val.shape)
    print(y_val.shape)

    X_train = [X1_train, X2_train, X3_train]
    X_val = [X1_val, X2_val, X3_val]

    model = lstm_with_shop_cate_embedding.get_model(hidden_size=20, input_seq_len=12, shop_num=50, cate_num=100, embed_size=2, output_dim=1)
    model = train.train_model(model, X_train, y_train, X_val, y_val, epochs=200)

    y_hats = infer.infer_testset(model, X_val, y_val)

    # y_test_invscaled = dataloader.get_invertscaled_values('Marcap', y_test)
    # y_hats_invscaled = dataloader.get_invertscaled_values('Marcap', y_hats)

    infer.draw_result_plot(y_hats, y_val)