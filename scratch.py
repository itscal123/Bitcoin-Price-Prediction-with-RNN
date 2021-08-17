import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from data import get_data_df
from sklearn.preprocessing import MinMaxScaler

df = get_data_df()
test_set = df.tail(100)
df = df.drop(100)
train_set = df.iloc[:, 3:4].values


scaler = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = scaler.fit_transform(train_set)

X_train = []
y_train = []
for i in range(30, 851):
    X_train.append(training_set_scaled[i-30:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = keras.Sequential([
    keras.layers.LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1), dropout=0.2),
    keras.layers.LSTM(units=32, return_sequences=True, dropout=0.2),
    keras.layers.LSTM(units=32, dropout=0.2),
    keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer="rmsprop", metrics=["mae"])
history = model.fit(X_train, y_train, epochs=20, batch_size=16)

test_prices = test_set.iloc[:, 3:4].values

data = pd.concat((df['Close (USD)'], test_set['Close (USD)']), axis = 0)
inputs = data[len(data) - len(test_set) - 30:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(30, 130):
    X_test.append(inputs[i-30:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plt.plot(test_prices, color = 'red', label = 'Real TATA Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted TAT Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()