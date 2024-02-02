import numpy as np
import matplotlib.pyplot as plt

# Define the linear regression model
class LinearRegression:
    def __init__(self):
        self.w = None
    
    def forward(self, x):
        return np.dot(x, self.w)
    
    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, x, y_pred, y_true):
        return np.dot(x.T, y_pred - y_true) / len(x)

# Baseline SGD
def sgd_update(model, gradient, learning_rate):
    model.w -= learning_rate * gradient

# Momentum update
def momentum_update(model, gradient, learning_rate, momentum, velocity):
    velocity = momentum * velocity - learning_rate * gradient
    model.w += velocity

# Adam update
def adam_update(model, gradient, learning_rate, beta1, beta2, epsilon, m, v, t):
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (gradient ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    model.w -= (learning_rate / (np.sqrt(v_hat) + epsilon)) * m_hat
    return m, v

# Define the training data generation function
def create_toy_data(func, sample_size, std, domain=None):
    if domain is None:
        domain = [0, 1]
    x = np.linspace(domain[0], domain[1], sample_size)
    np.random.shuffle(x)
    y = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, y

def sinusoidal(x):
    return np.sin(2 * np.pi * x)

# Generate training data
x_train, y_train = create_toy_data(sinusoidal, 10, 0.25)

# Define hyperparameters
learning_rate = 0.01
momentum = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
epochs = 1000

# Initialize linear regression model
model = LinearRegression()
model.w = np.random.randn(1)  # Initialize weights randomly

# Training loop for Baseline SGD
loss_history_sgd = []
for epoch in range(epochs):
    for x, y_true in zip(x_train, y_train):
        x = np.array([x])  # Convert x to a 2D array
        y_pred = model.forward(x)
        gradient = model.backward(x, y_pred, y_true)
        sgd_update(model, gradient, learning_rate)
    y_pred_epoch = model.forward(x_train[:, np.newaxis])
    loss_epoch = model.compute_loss(y_pred_epoch, y_train)
    loss_history_sgd.append(loss_epoch)

# Initialize linear regression model for Momentum
model = LinearRegression()
model.w = np.random.randn(1)  # Initialize weights randomly
velocity = 0

# Training loop for Momentum
loss_history_momentum = []
for epoch in range(epochs):
    for x, y_true in zip(x_train, y_train):
        x = np.array([x])  # Convert x to a 2D array
        y_pred = model.forward(x)
        gradient = model.backward(x, y_pred, y_true)
        momentum_update(model, gradient, learning_rate, momentum, velocity)
    y_pred_epoch = model.forward(x_train[:, np.newaxis])
    loss_epoch = model.compute_loss(y_pred_epoch, y_train)
    loss_history_momentum.append(loss_epoch)

# Initialize linear regression model for Adam
model = LinearRegression()
model.w = np.random.randn(1)  # Initialize weights randomly
m = 0
v = 0

# Training loop for Adam
loss_history_adam = []
for epoch in range(epochs):
    for x, y_true in zip(x_train, y_train):
        x = np.array([x])  # Convert x to a 2D array
        y_pred = model.forward(x)
        gradient = model.backward(x, y_pred, y_true)
        m, v = adam_update(model, gradient, learning_rate, beta1, beta2, epsilon, m, v, epoch + 1)
    y_pred_epoch = model.forward(x_train[:, np.newaxis])
    loss_epoch = model.compute_loss(y_pred_epoch, y_train)
    loss_history_adam.append(loss_epoch)



# Plot loss versus epoch for Momentum
plt.figure(figsize=(10, 5))
plt.plot(loss_history_sgd, label='SGD')
plt.plot(loss_history_momentum, label='Momentum', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Momentum: Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Plot loss versus epoch for Adam
plt.figure(figsize=(10, 5))
plt.plot(loss_history_sgd, label='SGD')
plt.plot(loss_history_adam, label='Adam', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Adam: Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.show()

