import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
import tensorflow as tf


def get_data(file_path: str, seq_size=1):
    data = pd.read_csv(file_path)
    data_de = data.drop(['end'], axis=1)

    # Format 'Date' column
    # ************************************************************************************
    data_de['start'] = pd.to_datetime(data_de.start)

    # Format load data into hourly data
    # ************************************************************************************
    data_de = data_de.set_index('start')
    data_de = data_de.groupby(pd.Grouper(freq='h')).sum()
    data_de.reset_index(inplace=True)

    # Split data into training and testing sets
    # ************************************************************************************
    train = data_de.loc[data_de['start'] <= '2019-05-31 00:00']
    test = data_de.loc[data_de['start'] > '2019-05-31 00:00']

    # train = train.iloc[:, 1:2].values
    # test = test.iloc[:, 1:2].values

    # Preview the data
    # ************************************************************************************
    # plt.figure(figsize=(16, 10))
    # plt.plot(data_de.index, data_de['load'])
    # plt.ylabel("load (mW)", fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.title('Germany Power Consumption(2015-2020) ', fontsize=20)
    # plt.show()

    # Standardize the data
    # ************************************************************************************
    scaler = StandardScaler()
    # scaler = MinMaxScaler(feature_range=(0, 1))

    # train['load_std'] = scaler.fit_transform(train[['load']]).flatten()
    # test['load_std'] = scaler.fit_transform(test[['load']]).flatten()
    train.loc[:, 'load'] = scaler.fit_transform(train[['load']]).flatten()
    test.loc[:, 'load'] = scaler.fit_transform(test[['load']]).flatten()

    # Get sequence data:
    # ************************************************************************************
    x_train, y_train = to_sequences(train[['load']], train['load'], seq_size)
    x_test, y_test = to_sequences(test[['load']], test['load'], seq_size)

    return x_train, y_train, x_test, y_test


def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []
    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])
    return np.array(x_values), np.array(y_values)


def plot_accuracy_and_loss(history, epochs: int = 25):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
