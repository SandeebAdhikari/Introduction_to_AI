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
        gradient = np.dot(x.T, y_pred - y_true) / len(x)
        return gradient
    
    def update_parameters(self, gradient, learning_rate):
        self.w -= learning_rate * gradient

# Define the training data generation function
def create_toy_data(func, sample_size, std, domain=[0, 1]):
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
epochs = 1000

# Initialize linear regression model
model = LinearRegression()
model.w = np.random.randn(1)  # Initialize weights randomly

# Training loop
loss_history = []
for epoch in range(epochs):
    # Shuffle the training data
    indices = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]
    
    # Iterate through each sample in the training data
    for x, y_true in zip(x_train_shuffled, y_train_shuffled):
        x = np.array([x])  # Convert x to a 2D array
        y_pred = model.forward(x)
        loss = model.compute_loss(y_pred, y_true)
        gradient = model.backward(x, y_pred, y_true)
        model.update_parameters(gradient, learning_rate)
    
    # Compute loss at the end of each epoch
    y_pred_epoch = model.forward(x_train[:, np.newaxis])
    loss_epoch = model.compute_loss(y_pred_epoch, y_train)
    loss_history.append(loss_epoch)

# Plot loss versus epoch
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.grid(True)
plt.show()
