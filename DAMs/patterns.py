import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def generate_pattern(size=20, correlation_length=1.0):
    """Generate a binary pattern with spatial correlations."""
    noise = np.random.randn(size, size)
    smooth = gaussian_filter(noise, sigma=correlation_length)
    return np.sign(smooth - np.median(smooth))


def generate_patterns(n_patterns, size=20, correlation_length=1.0):
    """Generate multiple patterns."""
    return np.array([generate_pattern(size, correlation_length) for _ in range(n_patterns)])


def show_patterns(patterns, title=None):
    """Display patterns in a row."""
    n = len(patterns)
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, patterns):
        ax.imshow(p, cmap='gray', vmin=-1, vmax=1)
        ax.axis('off')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Demo: show patterns at different correlation lengths
    for xi in [0,0.5,1, 2, 5,10,20]:
        patterns = generate_patterns(5, size=30, correlation_length=xi)
        show_patterns(patterns, title=f'ξ = {xi}')