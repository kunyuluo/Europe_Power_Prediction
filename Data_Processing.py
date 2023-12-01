import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
import tensorflow as tf


data = pd.read_csv(r'datasets/de.csv')
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
scaler.fit(train[['load']])

train[['load']] = scaler.fit_transform(train[['load']])
test[['load']] = scaler.transform(test[['load']])

# Get sequence data:
# ************************************************************************************
last_n = 24


def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []
    for i in range(len(x)-seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
    return np.array(x_values), np.array(y_values)


x_train, y_train = to_sequences(train[['load']], train['load'], last_n)
x_test, y_test = to_sequences(test[['load']], test['load'], last_n)

print(x_train)
print(y_train)
