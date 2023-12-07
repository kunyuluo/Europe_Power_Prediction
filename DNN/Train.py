from Data_Processing import get_data, plot_accuracy_and_loss
from DNN.DNN_Model import dnn_model
import pickle
import json

# Load the data
# ************************************************************************************
data_path = '../datasets/de.csv'
x_train, y_train, x_test, y_test, scaler = get_data(data_path, seq_size=48)
# print(x_train)

# Train the model
# ************************************************************************************
model, history = dnn_model(x_train, y_train, x_test, y_test, seq_size=48)

plot_accuracy_and_loss(history)

# Save the model
# ************************************************************************************
with open('dnn_model', 'wb') as file:
    pickle.dump(model, file)
    print('Model saved')

# saved_model = model.to_json()
# with open('dnn_model.json', 'w') as file:
#     json.dump(saved_model, file)
#     print('Model saved')
