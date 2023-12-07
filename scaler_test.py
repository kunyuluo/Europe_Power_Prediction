import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# dataset = load_iris()
# data_x = dataset.data
# data_y = dataset.target

data = pd.DataFrame([[1, 40, 700], [2, 50, 800], [3, 60, 900]])
print(data)

scaler1 = StandardScaler()
scaler2 = MinMaxScaler(feature_range=(0, 1))
data_x = scaler2.fit_transform(data)
data_x = pd.DataFrame(data_x)
print(data_x)
