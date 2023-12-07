import tensorflow as tf


# LSTM Model:
# ************************************************************************************
def lstm_model(x_train, y_train, x_test, y_test, epochs: int = 25, batch_size: int = 32):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.LSTM(32))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                        verbose=2)
    return model, history


def lstm_predict(model, x_test, scaler):
    if len(x_test.shape) == 3:
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    predict_scaled = model.predict(x_test)
    predict_unscaled = scaler.inverse_transform(predict_scaled)

    return predict_unscaled
