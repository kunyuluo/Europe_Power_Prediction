import tensorflow as tf


# DNN Model:
# ************************************************************************************
def dnn_model(x_train, y_train, x_test, y_test, epochs: int = 25, batch_size: int = 32, seq_size: int = 48):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, input_dim=seq_size, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                        verbose=2)
    return model, history


def dnn_predict(model, x_test, scaler):
    predict_scaled = model.predict(x_test)
    predict_unscaled = scaler.inverse_transform(predict_scaled)

    return predict_unscaled
