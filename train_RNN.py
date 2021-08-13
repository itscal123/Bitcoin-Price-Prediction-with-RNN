import tensorflow as tf
from tensorflow import keras
from preprocess_data import get_data
from utils import plot_multiple_forecasts


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


if __name__ == "__main__":
    train, test = get_data()
    model = keras.models.Sequential([
        keras.layers.LSTM(20, return_sequences=True),
        keras.layers.LSTM(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(3))
    ])
    model.compile(loss="mae", optimizer="nadam")
    history = model.fit(train, epochs=10)
    model.save("saved_model\my_model")  
