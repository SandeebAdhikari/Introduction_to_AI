import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define parameters for the second bivariate normal distribution
mu1 = np.array([2, 0])
sigma1 = np.array([[0.5, -1],
                  [-1, 6]])

# Create multivariate normal distributions
gaussian_dist1 = multivariate_normal(mean=mu1, cov=sigma1)

# Generate random samples from the distributions
samples1 = gaussian_dist1.rvs(size=200)

# Plot the contours of the distributions
x, y = np.mgrid[-4:8:.01, -8:8:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

plt.figure(figsize=(10, 8))

#Plot contours for the second distribution
plt.contour(x, y, gaussian_dist1.pdf(pos), colors='r', alpha=0.5)

# Scatter plot for simulated points

plt.scatter(samples1[:, 0], samples1[:, 1], color='b', label='Distribution 1 Sampled Points')

plt.title('Contours of Bivariate Gaussian Distributions with Simulated Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()