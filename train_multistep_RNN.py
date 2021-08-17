import tensorflow as tf
from window import WindowGenerator
from baseline import MultiStepBaseline
import numpy as np
import matplotlib.pyplot as plt
from data import Data, get_data_df


"""
Train model to predict next 3 days of stock prices, given 7 days of the past
"""

class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units 
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Wrap the LSTMCell in an RNN to simplify the warmup method
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)


    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state


    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                    training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


def compile_and_fit(model, window, patience=8):
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


if __name__ == "__main__":
    OUT_STEPS = 3
    MAX_EPOCHS = 50
    data = Data(get_data_df())
    train_df, val_df, test_df, num_features = data.get_data()
    multi_window = WindowGenerator(input_width=7, 
                                label_width=OUT_STEPS, 
                                shift=OUT_STEPS, 
                                train_df=train_df, 
                                val_df=val_df, 
                                test_df=test_df)

    feedback_model = FeedBack(32, out_steps=OUT_STEPS, num_features=num_features)
    prediction, state = feedback_model.warmup(multi_window.example[0])
    print(prediction.shape)
    print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)
    history = compile_and_fit(feedback_model, multi_window)

    multi_window.plot(feedback_model)