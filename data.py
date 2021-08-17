import tensorflow as tf
import pandas as pd


def get_data_df(path="data/data.csv"):
    df = pd.read_csv(path) 
    df = df.iloc[:,1:]
    return df


class Data():
    def __init__(self, df, train=0.8, valid=0.1, test=0.1):
        column_indices = {name : i for i, name in enumerate(df.columns)}
        n = len(df)
        train_df = df[0:int(n*train)]
        val_df = df[int(n*train):int(n*(1-test))]
        test_df = df[int(n*(1-test)):]
        num_features = df.shape[1]

        self.train = train_df
        self.valid = val_df
        self.test = test_df
        self.num_features = num_features

    def normalize(self):
        self.train_mean = self.train.mean()
        self.train_std = self.train.std()

        self.train = (self.train - self.train_mean) / self.train_std
        self.valid = (self.valid - self.train_mean) / self.train_std
        self.test = (self.test - self.train_mean) / self.train_std

    def get_data(self):
        self.normalize()
        return self.train, self.valid, self.test, self.num_features