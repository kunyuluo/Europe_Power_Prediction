import pickle
from Data_Processing import get_data
from sklearn.preprocessing import StandardScaler

with open('model', 'rb') as f:
    model = pickle.load(f)

# Load the data
# ************************************************************************************
data_path = 'datasets\\de.csv'
x_train, y_train, x_test, y_test = get_data(data_path, seq_size=48)
predict_scaled = model.predict(x_test)

# scaler = StandardScaler()
# predict_unscaled = scaler.inverse_transform(predict_scaled)

print(predict_scaled)
