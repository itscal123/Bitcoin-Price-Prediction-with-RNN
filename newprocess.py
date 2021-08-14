from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm._tqdm_notebook import tqdm_notebook as tqdm

def preprocess(test_size=0.2):
    """
    Reads the data.csv in the data folder, drops the timestamp column (1st column),
    splits the data into a train and validation set, normalizes the data, and
    returns the mutated data. Note the we use a default test_size of 0.2, but this 
    can be easily replaced. Since we are using time series, we don't shuffle the data
    params: None
    returns: X_train, X_test, y_train, y_test (tuple of numpy.ndarrays)
    """
    df = pd.read_csv("data/data.csv")
    df = df.iloc[:,1:]
    X, y = df.loc[:, df.columns != 'Close (USD)'], df["Close (USD)"]
    return X, y

def build_training_data(y_col_index, time_steps):
    df = pd.read_csv("data/data.csv")
    df = df.iloc[:,1:]
    dim_0 = df.shape[0] - time_steps
    dim_1 = df.shape[1]
    x = np.zeros((dim_0, time_steps, dim_1))
    y = np.zeros((x.shape[0],))

    for i in tqdm(range(dim_0)):
        x[i] = df[i:time_steps + i]
        y[i] = df[time_steps + i, y_col_index]
    print("length of time-series i/o {} {}".format(x.shape, y.shape))
    return x, y


if __name__ == "__main__":
    df = pd.read_csv("data/data.csv")
    df = df.iloc[:,1:]
    train_close = df.iloc[:, 3:4].values
    scaler = MinMaxScaler()
    train_close_scaled = scaler.fit_transform(train_close)

    X_train, y_train = [], []
    for i in range(20, len(train_close_scaled)):
        X_train.append(train_close_scaled[i-20:i,0])
        print(train_close_scaled[i-20:i,0])
        y_train.append(train_close_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
    print(X_train)