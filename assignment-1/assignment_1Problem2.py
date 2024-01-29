import numpy as np
import plotly.graph_objects as go

# Step 1: Simulate a 3-dimensional Gaussian random vector
covariance_matrix = np.array([[4, 2, 1],
                              [2, 3, 1.5],
                              [1, 1.5, 2]])

# Step 2: Singular Value Decomposition (SVD)
U, _, _ = np.linalg.svd(covariance_matrix)

# Step 3: Simulate random vectors and project onto subspace
num_samples = 1000
X = np.random.normal(0, 1, size=(num_samples, 3))
Y = X.dot(U)

# Project onto the subspace spanned by the first two principal components
U_proj = U[:, :2]
projection = U_proj.T.dot(Y.T).T

# Step 4: Visualize using Plotly
fig = go.Figure()

# Scatter plot for original data
fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers', name='Original Data'))

# Scatter plot for projected data
fig.add_trace(go.Scatter3d(x=projection[:, 0], y=projection[:, 1], mode='markers', name='Projected Data'))

# Update layout for better visualization
fig.update_layout(scene=dict(aspectmode="cube", xaxis=dict(title='X axis'), yaxis=dict(title='Y axis'), zaxis=dict(title='Z axis')))

# Display correlation matrices on the side
correlation_matrix_original = np.corrcoef(X, rowvar=False)
correlation_matrix_projected = np.corrcoef(projection, rowvar=False)

print("Correlation Matrix - Original Data:")
print(correlation_matrix_original)

print("\nCorrelation Matrix - Projected Data:")
print(correlation_matrix_projected)

# Show the plot
fig.show()
