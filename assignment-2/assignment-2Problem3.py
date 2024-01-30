import numpy as np
import matplotlib.pyplot as plt

def generate_linear_data(num_samples=100, noise=0.1):
    np.random.seed(42)
    X = 2 * np.random.rand(num_samples, 1)
    y = 4 + 3 * X + np.random.randn(num_samples, 1) * noise
    return X, y

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

def stochastic_gradient_descent(X, y, theta, learning_rate=0.01, epochs=100):
    m = len(y)
    cost_history = np.zeros(epochs)
    
    for epoch in range(epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            X_i = X[random_index].reshape(1, -1)
            y_i = y[random_index].reshape(1, -1)
            prediction = np.dot(X_i, theta)
            theta = theta - (1/m) * learning_rate * (X_i.T.dot((prediction - y_i)))
        cost_history[epoch] = compute_cost(X, y, theta)
    return theta, cost_history

# Generate some linear data
X, y = generate_linear_data()

# Hyperparameters
learning_rate = 0.01
epochs = 100

# Initial theta
theta = np.random.randn(2, 1)

# Add bias term to X
X_b = np.c_[np.ones((len(X), 1)), X]

# Run stochastic gradient descent
theta_final, cost_history = stochastic_gradient_descent(X_b, y, theta, learning_rate, epochs)

# Plot loss vs epoch
plt.plot(range(epochs), cost_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.show()
