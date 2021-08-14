import tensorflow as tf
from setup_data import retrieve_data
from window import WindowGenerator
from baseline import Baseline
import numpy as np
import matplotlib.pyplot as plt

MAX_EPOCHS = 30

def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


def compile_and_fit(model, window, patience=10):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=patience,
                                                      mode="min",
                                                      restore_best_weights=True)

    model.compile(loss=tf.losses.MeanSquaredError(), 
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])

    return history

# Multistep single output LSTM
simple_lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3))
])

if __name__ == "__main__":
    train_df, val_df, test_df = retrieve_data()
    single_output_window = WindowGenerator(
        input_width=7, label_width=7, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['Close (USD)'])
    print('Input shape:', single_output_window.example[0].shape)
    print('Output shape:', simple_lstm_model(single_output_window.example[0]).shape)
    history = compile_and_fit(simple_lstm_model, single_output_window)
    val_performance, performance = {}, {}
    val_performance["LSTM"] = simple_lstm_model.evaluate(single_output_window.val)
    performance["LSTM"] = simple_lstm_model.evaluate(single_output_window.test, verbose=0)
    single_output_window.plot(simple_lstm_model)

    # Performance
    x = np.arange(len(performance))
    width = 0.3
    metric_name = 'mean_absolute_error'
    metric_index = simple_lstm_model.metrics_names.index('mean_absolute_error')
    val_mae = [v[metric_index] for v in val_performance.values()]
    test_mae = [v[metric_index] for v in performance.values()]

    plt.ylabel('mean_absolute_error [T (degC), normalized]')
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=performance.keys(),
            rotation=45)
    _ = plt.legend()
    plt.show()