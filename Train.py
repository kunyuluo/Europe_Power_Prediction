from Data_Processing import get_data, plot_accuracy_and_loss
from LSTM_Model import lstm_model, lstm_predict
import pickle
import json

# Load the data
# ************************************************************************************
data_path = 'datasets\\de.csv'
x_train, y_train, x_test, y_test = get_data(data_path, seq_size=48)
# print(x_train)

# Train the model
# ************************************************************************************
model, history = lstm_model(x_train, y_train, x_test, y_test)

plot_accuracy_and_loss(history)

# Save the model
# ************************************************************************************
with open('model', 'wb') as file:
    pickle.dump(model, file)
    print('Model saved')

with open('scaler', 'wb') as file:
    pickle.dump(model, file)
    print('Scaler saved')
