import numpy as np
import pandas as pd


class DataLoader():

    def __init__(self):
        self.init_data()


    def init_data(self):
        item_dataset_filepath = 'dataset/items.csv'
        self.df_item = pd.read_csv(item_dataset_filepath)

        cate_dataset_filepath = 'dataset/item_categories.csv'
        self.df_item_category = pd.read_csv(cate_dataset_filepath)

        train_dataset_filepath = 'dataset/sales_train.csv.gz'
        df_train = pd.read_csv(train_dataset_filepath, parse_dates=['date'], date_parser=dateparse)
        self.df_train = pd.merge(df_train, self.df_item)



    def num_shop(self):
        return len(self.user2index)

    def num_item(self):
        return len(self.movie2index)

    def num_item_category(self):
        return len(self.movie2index)

    def get_movie_info(self, movie_list):
        return self.df_movies.loc[movie_list]

    def load_data(self, x_seq_len=12, train_ratio=0.8):

        grouped_item_cnt = self.df_train['item_cnt_day'].groupby(
            [self.df_train['shop_id'], self.df_train['item_category_id'], self.df_train['date_block_num']]).sum()
        grouped_item_cnt_frame = grouped_item_cnt.to_frame()
        total_date_block_num = len(grouped_item_cnt_frame['date_block_num'].unique())
        grouped_shop_category = grouped_item_cnt_frame.groupby(['shop_id', 'item_category_id']).size().reset_index()

        grouped_item_cnt_frame_pv = pd.pivot_table(
            grouped_item_cnt_frame,
            columns=['shop_id', 'item_category_id'],
            index='date_block_num',
            values='item_cnt_day',
            fill_value=0)


        # train_data = [(x1, x2, x3, y)]
        dataset = [( shop_id,
                    category_id,
                    [grouped_item_cnt_frame_pv[shop_id][category_id][i] for i in range(j, j+x_seq_len)],
                    grouped_item_cnt_frame_pv[shop_id][category_id][j+x_seq_len] )
                 for j in range(total_date_block_num-x_seq_len)
                 for shop_id, category_id in grouped_shop_category[['shop_id', 'item_category_id']].values]

        # split into train/test set
        msk = np.random.rand(len(dataset)) < train_ratio
        train_data = dataset[msk]
        val_data = dataset[~msk]

        X1_train = np.array([X1 for (X1, X2, X3, y) in train_data])
        X2_train = np.array([X2 for (X1, X2, X3, y) in train_data])
        X3_train = np.array([X3 for (X1, X2, X3, y) in train_data])
        y_train = np.array([y for (X1, X2, X3, y) in train_data])

        X1_val = np.array([X1 for (X1, X2, X3, y) in val_data])
        X2_val = np.array([X2 for (X1, X2, X3, y) in val_data])
        X3_val = np.array([X3 for (X1, X2, X3, y) in val_data])
        y_val = np.array([y for (X1, X2, X3, y) in val_data])


        return  (X1_train, X2_train, X3_train, y_train), (X1_val, X2_val, X3_val, y_val)

