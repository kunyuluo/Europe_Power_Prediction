import pickle
import matplotlib.pyplot as plt
from Data_Processing import get_data, evaluate_accuracy

with open('model', 'rb') as f:
    model = pickle.load(f)

# Load the data
# ************************************************************************************
data_path = '../datasets/de.csv'
x_train, y_train, x_test, y_test, scaler = get_data(data_path, seq_size=48)
predict_scaled = model.predict(x_test)
predict_unscaled = scaler.inverse_transform(predict_scaled).flatten()

real_power = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate accuracy of test dataset
# ************************************************************************************
result = evaluate_accuracy(real_power, predict_unscaled)
print('Prediction Accuracy is {}'.format(result))

# Visualize the predicted results
# ************************************************************************************

plt.plot(real_power, color='black', label='History Power')
plt.plot(predict_unscaled, color='green', label='Predicted Power')
plt.title('Elec Power Prediction')
plt.xlabel('Hour')
plt.ylabel('kW')
plt.legend()
plt.show()
