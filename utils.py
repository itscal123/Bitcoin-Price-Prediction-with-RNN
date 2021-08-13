import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from preprocess_data import get_data


def plot_multiple_forecasts(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0])
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "ro-", label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "bx-", label="Forecast", markersize=10)
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)

def plot_forecasts():
    df = pd.read_csv("data/data.csv")
    df = df.iloc[:,1:]
    y = df["Close (USD)"]
    y_train = pd.DataFrame(y.head(int(len(y)*(0.8))))
    y_test = pd.DataFrame(y.tail(int(len(y)*(0.2))))
    ax = y_train.plot()
    y_test.plot(ax=ax)
    plt.show()

if __name__ == "__main__":
    _, test = get_data()
    model = tf.keras.models.load_model("saved_model\my_model")
    print(model.evaluate(test))
    predictions = model.predict(test)
    print(predictions[0])

    """
    for batch in test.take(1):
        for arr in batch:
            print(arr.numpy())
    predictions =  model.predict(test)

    batch = None
    for i in test.take(1):
        batch = i[0].numpy()

    single = model.predict(batch)
    print(single[0][0])

    print(predictions[0][0])
    """
