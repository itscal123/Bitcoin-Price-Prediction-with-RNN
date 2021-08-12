from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pandas as pd


def preprocess():
    """
    Reads the data.csv in the data folder, drops the timestamp column (1st column),
    normalizes the data, and returns the mutated data.
    params: None
    returns: numpy.ndarray (n_samples, n_features_new)
    """
    df = pd.read_csv("data/data.csv")
    df = df.iloc[:,1:]
    data = MinMaxScaler().fit_transform(df)
    return data
