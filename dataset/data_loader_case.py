import numpy as np
import pandas as pd
import os
import pickle
from dataset.data_loader import DataLoader
from util.scaler import LogMinMaxScaler

DATASET_RAW_DIR = 'dataset/kaggle_raw/'

CASE_SALES_NONE=0
CASE_SALES_RARE=1
CASE_SALES_MANY=2

class CaseDataLoader(DataLoader):


    def get_cate_from_item_id(self, item_id):
        return self.df_item.loc[self.df_item.index[self.df_item['item_id'] == item_id][0], 'item_category_id']

    def get_shop_item_case(self, shop_id, item_id, threshhold=12):
        try:
            trade_month_count = self.train_case_pv[shop_id][item_id]
            if trade_month_count <= 0:
                return CASE_SALES_NONE
            elif trade_month_count <= threshhold:
                return CASE_SALES_RARE
            else:
                return CASE_SALES_MANY

        except Exception as e:
            return CASE_SALES_NONE


    def load_data(self, x_seq_len=12, train_ratio=0.8, load_pickle=True, threshhold=12):

        if load_pickle and os.path.exists(self.dataset_pickle_path):
            with open(self.dataset_pickle_path, 'rb') as file:
                dataset = pickle.load(file)
                dataset1, dataset2 = dataset
        else:
            train_dataset_filepath = DATASET_RAW_DIR + 'sales_train.csv.gz'
            dateparse = lambda dates: pd.datetime.strptime(dates, '%d.%m.%Y')
            df_train = pd.read_csv(train_dataset_filepath, parse_dates=['date'], date_parser=dateparse)
            self.df_train = pd.merge(df_train, self.df_item)

            grouped_item_cnt = self.df_train['item_cnt_day'].groupby(
                [self.df_train['shop_id'], self.df_train['item_id'], self.df_train['date_block_num']]).sum()
            grouped_item_cnt_frame = grouped_item_cnt.to_frame()
            grouped_item_cnt_frame.reset_index(inplace=True)
            total_date_block_num = len(grouped_item_cnt_frame['date_block_num'].unique())
            grouped_shop_item = grouped_item_cnt_frame.groupby(['shop_id', 'item_id']).size()
            # grouped_shop_item_frame = grouped_shop_item.reset_index()

            self.grouped_item_cnt_frame_pv = pd.pivot_table(
                grouped_item_cnt_frame,
                columns=['shop_id', 'item_id'],
                index='date_block_num',
                values='item_cnt_day',
                fill_value=0)

            self.train_case_pv = pd.pivot_table(
                grouped_item_cnt_frame,
                columns='shop_id',
                index='item_id',
                values='item_cnt_day',
                aggfunc=np.size,
                fill_value=0)

            grouped_price_mean = self.df_train['item_price'].groupby(
                [self.df_train['shop_id'], self.df_train['item_id'], self.df_train['date_block_num']]).mean()

            # train_data = [(x1, x2, x3, y)]
            # x3.shape = (None, 12, 1)
            print('Making Dataset for CASE_SALES_RARE..')
            dataset1 = [([self.grouped_item_cnt_frame_pv[shop_id][item_id][i]
                         for i in range(j, j + x_seq_len)],
                         self.grouped_item_cnt_frame_pv[shop_id][item_id][j + x_seq_len])
                        for j in range(total_date_block_num - x_seq_len)
                        for shop_id, item_id in
                            grouped_shop_item[grouped_shop_item <= threshhold].reset_index()[['shop_id', 'item_id']].values]


            print('Making Dataset for CASE_SALES_MANY..')
            dataset2 = [(shop_id,
                        self.get_cate_from_item_id(item_id),
                        [[self.grouped_item_cnt_frame_pv[shop_id][item_id][i]#,grouped_price_mean[shop_id][item_id][i]  # 가격은 잠시생략...
                          ] for i in range(j, j + x_seq_len)],
                        self.grouped_item_cnt_frame_pv[shop_id][item_id][j + x_seq_len])
                    for j in range(total_date_block_num - x_seq_len)
                    for shop_id, item_id in
                        grouped_shop_item[grouped_shop_item > threshhold].reset_index()[['shop_id', 'item_id']].values]

            dataset = (dataset1, dataset2)

            with open(self.dataset_pickle_path, 'wb') as file:
                pickle.dump(dataset, file)

        print('total %d dataset1 loaded..' % len(dataset1))
        print('total %d dataset2 loaded..' % len(dataset2))

        msk1 = np.random.rand(len(dataset1)) < train_ratio
        msk2 = np.random.rand(len(dataset2)) < train_ratio

        X1 = np.array([X for (X, y) in dataset1])
        y1 = np.array([y for (X, y) in dataset1])

        X21 = np.array([X1 for (X1, X2, X3, y) in dataset1])
        X22 = np.array([X2 for (X1, X2, X3, y) in dataset1])
        X23 = np.array([X3 for (X1, X2, X3, y) in dataset1])
        y2 = np.array([y for (X1, X2, X3, y) in dataset1])


        # normalize X3, y data here..
        min_val = np.min([X23.min(), y2.min()])
        max_val = np.max([X23.max(), y2.max()])
        print('min_val :', min_val)
        print('max_val :', max_val)

        self.scaler = LogMinMaxScaler(min_val, max_val)
        print('scaler.log_minval:', self.scaler.min_logvalue)
        print('scaler.log_maxval:', self.scaler.max_logvalue)
        scale_func = lambda x : self.scaler.scale_value(x)
        X23 = scale_func(X23)
        y2 = scale_func(y2)

        print(X1[0])
        print(X23[0])

        X1_train = X1[msk1]
        y1_train = y1[msk1]

        X1_val = X1[~msk1]
        y1_val = y1[~msk1]

        X21_train = X21[msk2]
        X22_train = X22[msk2]
        X23_train = X23[msk2]
        y2_train = y2[msk2]

        X21_val = X21[~msk2]
        X22_val = X22[~msk2]
        X23_val = X23[~msk2]
        y2_val = y2[~msk2]


        return (X1_train, y1_train), (X1_val, y1_val), (X21_train, X22_train, X23_train, y2_train), (X21_val, X22_val, X23_val, y2_val)