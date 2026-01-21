import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class DenseAssociativeMemory:
    def __init__(self, patterns, beta=5.0):
        """
        patterns: array of shape (p, N) where p = number of patterns, N = number of neurons
        beta: inverse temperature (higher = sharper retrieval)
        """
        self.patterns = patterns
        self.p, self.N = patterns.shape
        self.beta = beta

    def softmax(self, z):
        z = z - z.max()
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()

    def energy(self, state):
        """Compute energy of a state."""
        similarities = self.patterns @ state
        return -np.log(np.sum(np.exp(self.beta * similarities))) / self.beta

    def update(self, state, n_steps=100):
        """Asynchronous updates until convergence or max steps."""
        state = state.copy().astype(float)
        for _ in range(n_steps):
            old_state = state.copy()
            for i in np.random.permutation(self.N):
                similarities = self.patterns @ state
                weights = self.softmax(self.beta * similarities)
                state[i] = self.patterns[:, i] @ weights
            if np.allclose(state, old_state):
                break
        return state

    def overlap(self, state, pattern_idx):
        """Overlap between state and a stored pattern."""
        return (state @ self.patterns[pattern_idx]) / self.N


def test_retrieval(network, pattern_idx, noise_level=0.5):
    """Corrupt a pattern and try to retrieve it."""
    pattern = network.patterns[pattern_idx].copy()

    n_flip = int(noise_level * network.N)
    flip_idx = np.random.choice(network.N, n_flip, replace=False)
    pattern[flip_idx] *= -1

    retrieved = network.update(pattern)
    overlap = network.overlap(retrieved, pattern_idx)

    return retrieved, overlap


def update_with_history(network, state, n_steps=100):
    """Yields states after each spin update."""
    state = state.copy().astype(float)
    yield state.copy()
    for _ in range(n_steps):
        old_state = state.copy()
        for i in np.random.permutation(network.N):
            similarities = network.patterns @ state
            weights = network.softmax(network.beta * similarities)
            state[i] = network.patterns[:, i] @ weights
            yield state.copy()
        if np.allclose(state, old_state):
            break


def animate_retrieval_random(network, size=20, interval=50):
    """Start from random noise and show convergence alongside all memories."""
    p = network.p

    state = np.random.choice([-1, 1], size=network.N).astype(float)

    history = list(update_with_history(network, state))

    fig, axes = plt.subplots(2, p, figsize=(3 * p, 6))

    for i in range(p):
        axes[0, i].imshow(network.patterns[i].reshape(size, size), cmap='gray', vmin=-1, vmax=1)
        axes[0, i].set_title(f'Memory {i}')
        axes[0, i].axis('off')

    mid = p // 2
    for i in range(p):
        if i != mid:
            axes[1, i].axis('off')

    im = axes[1, mid].imshow(history[0].reshape(size, size), cmap='gray', vmin=-1, vmax=1)
    axes[1, mid].set_title('Step 0')
    axes[1, mid].axis('off')

    def update(frame):
        im.set_array(history[frame].reshape(size, size))
        overlaps = [network.overlap(history[frame], i) for i in range(p)]
        best = np.argmax(overlaps)
        axes[1, mid].set_title(f'Step {frame} (closest: {best}, overlap: {overlaps[best]:.2f})')
        return [im]

    anim = FuncAnimation(fig, update, frames=len(history), interval=interval, blit=False)
    plt.tight_layout()
    plt.show()
    return anim


if __name__ == '__main__':
    from hopfield.patterns import generate_patterns

    patterns_2d = generate_patterns(3, size=20, correlation_length=2)
    patterns_flat = patterns_2d.reshape(3, -1)
    net = DenseAssociativeMemory(patterns_flat, beta=.1)

    animate_retrieval_random(net, interval=10)
