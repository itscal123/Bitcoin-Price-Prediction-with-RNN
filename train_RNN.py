import tensorflow as tf
from tensorflow import keras
from preprocess_data import get_data
from utils import plot_multiple_forecasts


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


if __name__ == "__main__":
    train, test = get_data()
    model = keras.models.Sequential([
        keras.layers.LSTM(20, return_sequences=True, input_shape=[3,16]),
        keras.layers.LSTM(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(3))
    ])
    model.compile(loss="mean_squared_logarithmic_error", optimizer="nadam")
    history = model.fit(train, epochs=20, batch_size=3)
