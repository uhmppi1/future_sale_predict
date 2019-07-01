from dataset import data_loader_case as data_loader
from model import lstm_with_shop_cate_embedding, simple_dense
import train, infer

from numpy.random import seed

data_pickle_file_path = 'dataset/pickle/dataset_12step_shop_cate_case.pkl'

seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

if __name__ == "__main__":
    x_seq_len = 12
    train_ratio = 0.8

    dataloader = data_loader.DataLoader(data_pickle_file_path)

    (X1_train, y1_train), (X1_val, y1_val), (X21_train, X22_train, X23_train, y2_train), (X21_val, X22_val, X23_val, y2_val) \
        = dataloader.load_data(x_seq_len, train_ratio)

    print(X1_train.shape)
    print(y1_train.shape)
    print(X1_val.shape)
    print(y1_val.shape)

    print(X21_train.shape)
    print(X22_train.shape)
    print(X23_train.shape)
    print(y2_train.shape)
    print(X21_val.shape)
    print(X22_val.shape)
    print(X23_val.shape)
    print(y2_val.shape)

    X2_train = [X21_train, X22_train, X23_train]
    X2_val = [X21_val, X22_val, X23_val]

    shop_num = dataloader.num_shop()
    cate_num = dataloader.num_item_category()

    model1 = simple_dense.get_model(12, 1)
    model2 = train.train_model(model1, X1_train, y1_train, X1_val, y1_val, epochs=1)

    model2 = lstm_with_shop_cate_embedding.get_model(hidden_size=20, x3_shape=(12,1),
                shop_num=shop_num, cate_num=cate_num, embed_size=2, output_dim=1)
    model2 = train.train_model(model2, X2_train, y2_train, X2_val, y2_val, epochs=1)

    y1_hats = infer.infer_testset(model1, X1_val, y1_val)
    y2_hats = infer.infer_testset(model2, X2_val, y2_val)

    y2_val_invscaled = dataloader.get_invertscaled_values(y2_val)
    y2_hats_invscaled = dataloader.get_invertscaled_values(y2_hats)

    infer.print_result(y1_hats, y1_val, title='dense')
    infer.print_result(y2_hats_invscaled, y2_val_invscaled, title='lstm')