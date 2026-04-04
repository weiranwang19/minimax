import numpy as np
import matplotlib.pyplot as plt

def f(w):
    """
    Nonlinear convex objective function:
    Pseudo-Huber-like in w1 + strongly convex quadratic in w2
    """
    x, y = w[0], w[1]
    return np.sqrt(x**2 + 1.0) + 0.25 * x**2 + 10.0 * y**2

def grad_f(w):
    """Gradient of the nonlinear objective function."""
    x, y = w[0], w[1]
    # Derivative of sqrt(x^2+1) is x / sqrt(x^2+1)
    gx = x / np.sqrt(x**2 + 1.0) + 0.5 * x
    gy = 20.0 * y
    return np.array([gx, gy])

# --- Hyperparameter Setup ---
# For this function, the maximum curvature (Lipschitz constant L) is approx 20 (from the y term).
# The minimum curvature (strong convexity mu) approaches 0.5 as x -> infinity.
L = 20.0
mu = 0.5

# Step size and momentum based on global L and mu
alpha = 1.0 / L
beta = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))

def gradient_descent(w0, num_iters):
    """Runs standard Gradient Descent."""
    w = w0.copy()
    trajectory = [w.copy()]

    for _ in range(num_iters):
        w = w - alpha * grad_f(w)
        trajectory.append(w.copy())

    return np.array(trajectory)

def nesterov_accelerated_gradient(w0, num_iters):
    """Runs Nesterov's Accelerated Gradient descent."""
    x = w0.copy()
    y = w0.copy()
    trajectory = [x.copy()]

    for _ in range(num_iters):
        x_prev = x.copy()

        # 1. Gradient step at lookahead point
        x = y - alpha * grad_f(y)
        # 2. Extrapolation step
        y = x + beta * (x - x_prev)

        trajectory.append(x.copy())

    return np.array(trajectory)

# --- Problem Setup ---
w_init = np.array([12.0, 5.0])
iterations = 60

# Run both algorithms
gd_traj = gradient_descent(w_init, iterations)
nag_traj = nesterov_accelerated_gradient(w_init, iterations)

# --- Plotting ---
plt.figure(figsize=(12, 8))

# Create grid for contour plot
w1_range = np.linspace(-4, 14, 100)
w2_range = np.linspace(-6, 6, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)
# Compute Z values for the contour
Z = np.sqrt(W1**2 + 1.0) + 0.25 * W1**2 + 10.0 * W2**2

# Plot contours (using log space to handle the steep y-walls gracefully)
levels = np.logspace(-0.5, 3, 20)
plt.contour(W1, W2, Z, levels=levels, cmap='viridis', alpha=0.4)

# Plot trajectories
plt.plot(gd_traj[:, 0], gd_traj[:, 1], 'go-', label='Gradient Descent', markersize=4, linewidth=1.5, alpha=0.8)
plt.plot(nag_traj[:, 0], nag_traj[:, 1], 'bo-', label="Nesterov's Accelerated Gradient", markersize=4, linewidth=1.5, alpha=0.8)

# Mark start and end points
plt.plot(w_init[0], w_init[1], 'k*', markersize=15, label='Start')
plt.plot(0, 0, 'y*', markersize=15, label='Optimum (0,0)')

# Formatting
plt.title("Comparison on a Nonlinear Convex Function (Pseudo-Huber + Quadratic)")
plt.xlabel('$w_1$ (Nonlinear, flatter direction)')
plt.ylabel('$w_2$ (Quadratic, steep direction)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.axis('equal')
plt.tight_layout()
plt.show()