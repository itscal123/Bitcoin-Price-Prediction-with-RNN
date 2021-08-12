from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd


def timeseries_dataset_multistep(X, y, input_sequence_length, output_sequence_length, batch_size):
    """
    Converts X feature matrix and y output labels into a tensorflow.data.Dataset tensor. 
    Adjusts the size of the tensor depending on passed input_sequence length (i.e., number
    of days leading up to predictions), output_sequence_length (i.e., number of days to predict),
    and batch_size (i.e., number of batches)
    params: X (numpy.ndarray), y (numpy.ndarray), input_sequence_length (int), output_sequence_length (int),
            batch_size (int)
    returns: timeseries tensor (tf.data.Dataset)
    """
    def extract_output(l):
        return l[:output_sequence_length]
     
    feature_ds = tf.keras.preprocessing.timeseries_dataset_from_array(X, None, input_sequence_length, batch_size=1).unbatch()
    label_ds = tf.keras.preprocessing.timeseries_dataset_from_array(y, None, input_sequence_length, batch_size=1) \
        .skip(input_sequence_length) \
        .unbatch() \
        .map(extract_output)
         
    return tf.data.Dataset.zip((feature_ds, label_ds)).batch(batch_size, drop_remainder=True)


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

    normalizer = StandardScaler()
    X_norm_train = normalizer.fit_transform(X_train)
    X_norm_test = normalizer.transform(X_test)
    return X_norm_train, X_norm_test, y_train, y_test


def get_data():
    """
    Returns tuple of tensors: one for the training data, the other for the test data
    params: none
    retunrs: tuple of tensors (tf.data.Dataset)
    """
    X_train, X_test, y_train, y_test = preprocess()
    train = timeseries_dataset_multistep(X_train, y_train, input_sequence_length=3, output_sequence_length=3, batch_size=3)
    test = timeseries_dataset_multistep(X_test, y_test, input_sequence_length=3, output_sequence_length=3, batch_size=3)
    return train, test