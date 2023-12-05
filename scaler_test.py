from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler

dataset = load_iris()
data_x = dataset.data
data_y = dataset.target

scaler1 = StandardScaler()
scaler2 = MinMaxScaler(feature_range=(0, 1))
data_x = scaler2.fit_transform(data_x)
print(data_x)
