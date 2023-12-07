from Data_Processing import get_data, plot_accuracy_and_loss
from LSTM.LSTM_Model import lstm_model
import pickle

# Load the data
# ************************************************************************************
data_path = '../datasets/de.csv'
x_train, y_train, x_test, y_test, scaler = get_data(data_path, seq_size=48)
# print(x_train)

# Train the model
# ************************************************************************************
model, history = lstm_model(x_train, y_train, x_test, y_test)

plot_accuracy_and_loss(history)

# Save the model
# ************************************************************************************
with open('lstm_model', 'wb') as file:
    pickle.dump(model, file)
    print('Model saved')

# with open('scaler', 'wb') as file:
#     pickle.dump(scaler, file)
#     print('Scaler saved')
