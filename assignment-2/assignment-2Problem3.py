import numpy as np
import matplotlib.pyplot as plt

# Define the linear regression model
class LinearRegression:
    def __init__(self):
        self.w = None
    
    def forward(self, X):
        return np.dot(X, self.w)
    
    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, X, y_pred, y_true):
        gradient = np.dot(X.T, y_pred - y_true) / len(X)
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

# Transform input data to polynomial features
X_train = np.column_stack((x_train**0, x_train**1, x_train**2, x_train**3))
input_dim = X_train.shape[1]
model.w = np.random.randn(input_dim)  # Initialize weights randomly

# Training loop
loss_history = []
for epoch in range(epochs):
    # Shuffle the training data
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]
    
    # Iterate through each sample in the training data
    for X, y_true in zip(X_train_shuffled, y_train_shuffled):
        X = np.array([X])  # Convert X to a 2D array
        y_pred = model.forward(X)
        loss = model.compute_loss(y_pred, y_true)
        gradient = model.backward(X, y_pred, y_true)
        model.update_parameters(gradient, learning_rate)
    
    # Compute loss at the end of each epoch
    y_pred_epoch = model.forward(X_train)
    loss_epoch = model.compute_loss(y_pred_epoch, y_train)
    loss_history.append(loss_epoch)

# Plot loss versus epoch
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.grid(True)
plt.show()
