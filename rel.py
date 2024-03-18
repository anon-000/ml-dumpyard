from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


model = keras. Sequential([keras. layers. Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
model.fit(xs, ys, epochs=500)

prediction_input = np.array([10.0])
prediction_output = model.predict(prediction_input)
print(prediction_output)


# I want to plot
 
plt.scatter(xs, ys, color='blue', label='Original data')

# Plot the predicted output for the input [10.0]
plt.scatter([10.0], prediction_output, color='red', label='Prediction')

# Add labels and title
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Linear Regression Prediction')
plt.legend()

# Show the plot
plt.show()