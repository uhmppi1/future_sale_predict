import numpy as np
import pandas as pd
import os
import pickle

DATASET_RAW_DIR = 'dataset/kaggle_raw/'

class DataLoader():

    def __init__(self, pickle_file_path):
        self.init_data()
        self.dataset_pickle_path = pickle_file_path


    def init_data(self):
        shop_dataset_filepath = DATASET_RAW_DIR + 'shops.csv'
        self.df_shop = pd.read_csv(shop_dataset_filepath)

        item_dataset_filepath = DATASET_RAW_DIR + 'items.csv'
        self.df_item = pd.read_csv(item_dataset_filepath)

        cate_dataset_filepath = DATASET_RAW_DIR + 'item_categories.csv'
        self.df_item_category = pd.read_csv(cate_dataset_filepath)


    def num_shop(self):
        return len(self.df_shop)

    def num_item(self):
        return len(self.df_item)

    def num_item_category(self):
        return len(self.df_item_category)


    def item2cate(self, item_no):
        return self.df_item.loc[self.df_item.index[self.df_item['item_id'] == item_no][0], 'item_category_id']


    def load_data(self, x_seq_len=12, train_ratio=0.8, load_pickle=True):

        if load_pickle and os.path.exists(self.dataset_pickle_path):
            with open(self.dataset_pickle_path, 'rb') as file:
                dataset = pickle.load(file)
        else:
            train_dataset_filepath = DATASET_RAW_DIR + 'sales_train.csv.gz'
            dateparse = lambda dates: pd.datetime.strptime(dates, '%d.%m.%Y')
            df_train = pd.read_csv(train_dataset_filepath, parse_dates=['date'], date_parser=dateparse)
            self.df_train = pd.merge(df_train, self.df_item)

        grouped_item_cnt = self.df_train['item_cnt_day'].groupby(
            [self.df_train['shop_id'], self.df_train['item_category_id'], self.df_train['date_block_num']]).sum()
        grouped_item_cnt_frame = grouped_item_cnt.to_frame()
        grouped_item_cnt_frame.reset_index(inplace=True)
        total_date_block_num = len(grouped_item_cnt_frame['date_block_num'].unique())
        grouped_shop_category = grouped_item_cnt_frame.groupby(['shop_id', 'item_category_id']).size().reset_index()

        grouped_item_cnt_frame_pv = pd.pivot_table(
            grouped_item_cnt_frame,
            columns=['shop_id', 'item_category_id'],
            index='date_block_num',
            values='item_cnt_day',
            fill_value=0)

        # train_data = [(x1, x2, x3, y)]
        # x3.shape = (None, 12, 1)
        print('Making Dataset..')
        dataset = [( shop_id,
                    category_id,
                    [[grouped_item_cnt_frame_pv[shop_id][category_id][i]] for i in range(j, j+x_seq_len)],
                    grouped_item_cnt_frame_pv[shop_id][category_id][j+x_seq_len] )
                for j in range(total_date_block_num-x_seq_len)
                for shop_id, category_id in grouped_shop_category[['shop_id', 'item_category_id']].values]

        with open(self.dataset_pickle_path, 'wb') as file:
            pickle.dump(dataset, file)

        print('total %d dataset loaded..' % len(dataset))

        msk = np.random.rand(len(dataset)) < train_ratio

        X1 = np.array([X1 for (X1, X2, X3, y) in dataset])
        X2 = np.array([X2 for (X1, X2, X3, y) in dataset])
        X3 = np.array([X3 for (X1, X2, X3, y) in dataset])
        y = np.array([y for (X1, X2, X3, y) in dataset])

        X1_train = X1[msk]
        X2_train = X2[msk]
        X3_train = X3[msk]
        y_train = y[msk]

        X1_val = X1[~msk]
        X2_val = X2[~msk]
        X3_val = X3[~msk]
        y_val = y[~msk]

        # X1_train = np.array([X1 for (X1, X2, X3, y) in train_data])
        # X2_train = np.array([X2 for (X1, X2, X3, y) in train_data])
        # X3_train = np.array([X3 for (X1, X2, X3, y) in train_data])
        # y_train = np.array([y for (X1, X2, X3, y) in train_data])
        #
        # X1_val = np.array([X1 for (X1, X2, X3, y) in val_data])
        # X2_val = np.array([X2 for (X1, X2, X3, y) in val_data])
        # X3_val = np.array([X3 for (X1, X2, X3, y) in val_data])
        # y_val = np.array([y for (X1, X2, X3, y) in val_data])

        return (X1_train, X2_train, X3_train, y_train), (X1_val, X2_val, X3_val, y_val)