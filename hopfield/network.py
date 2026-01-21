import numpy as np
from scipy.ndimage import gaussian_filter


def generate_pattern(size, correlation_length):
    """Generate a single binary pattern with spatial correlations."""
    noise = np.random.randn(size, size)
    smooth = gaussian_filter(noise, sigma=correlation_length)
    return np.sign(smooth - np.median(smooth))


def generate_network(N, M, xi):
    """
    Create a Hopfield network with M memories on an NxN grid.

    Returns:
        patterns: array of shape (M, N*N) - the stored patterns (flattened)
        W: array of shape (N*N, N*N) - the weight matrix
    """
    # Generate patterns
    patterns = np.array([generate_pattern(N, xi).flatten() for _ in range(M)])

    # Build weight matrix
    n_neurons = N * N
    W = (patterns.T @ patterns) / n_neurons
    np.fill_diagonal(W, 0)

    return patterns, W