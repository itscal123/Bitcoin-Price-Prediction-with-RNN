from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd


def timeseries_dataset_one_step(features, labels, input_sequence_length, batch_size):
    return tf.keras.preprocessing.timeseries_dataset_from_array(features[:-1], \
            np.roll(labels, -input_sequence_length, axis=0)[:-1], \
            input_sequence_length, batch_size=batch_size)
 




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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    normalizer = MinMaxScaler()
    X_norm_train = normalizer.fit_transform(X_train)
    X_norm_test = normalizer.transform(X_test)
    return X_norm_train, X_norm_test, y_train, y_test


def get_data_df(path="data/data.csv"):
    """
    Returns dataframe of the data.csv in chronological order
    params: path to data.csv
    returns: dataframe
    """
    df = pd.read_csv(path)
    df = df.iloc[:,1:]
    return df


def split_data(df, train=0.8, valid=0.1, test=0.1):
    """
    Splits dataframe into train, valid, and test splits.
    params: df (DataFrame), train (float), valid (float), test (float)
    returns: tuple of dataframes and number of features
    """
    column_indices = {name : i for i, name in enumerate(df.columns)}
    n = len(df)
    train_df = df[0:int(n*train)]
    val_df = df[int(n*train):int(n*(1-test))]
    test_df = df[int(n*(1-test)):]

    num_features = df.shape[1]

    return train_df, val_df, test_df, num_features

def normalize(train_df, val_df, test_df):
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    return train_df, val_df, test_df

def retrieve_data():
    train_df, val_df, test_df, num_features = split_data(get_data_df())
    train_df, val_df, test_df = normalize(train_df, val_df, test_df)
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df, num_features = split_data(get_data_df())
    print(num_features)
    train_df, val_df, test_df = normalize(train_df, val_df, test_df)
    print(train_df)
    print(val_df)
    print(test_df)
    