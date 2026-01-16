import numpy as np

class HopfieldNetwork:
    def __init__(self, patterns):
        """
        patterns: array of shape (p, N) where p = number of patterns, N = number of neurons
                  (flatten your 2D images first)
        """
        self.patterns = patterns
        self.p, self.N = patterns.shape

        # Hebbian learning rule
        self.W = (patterns.T @ patterns) / self.N
        np.fill_diagonal(self.W, 0)

    def energy(self, state):
        """Compute energy of a state."""
        return -0.5 * state @ self.W @ state

    def update(self, state, n_steps=100):
        """Asynchronous updates until convergence or max steps."""
        state = state.copy()
        for _ in range(n_steps):
            old_state = state.copy()
            for i in np.random.permutation(self.N):
                h = self.W[i] @ state
                state[i] = np.sign(h) if h != 0 else state[i]
            if np.array_equal(state, old_state):
                break
        return state

    def overlap(self, state, pattern_idx):
        """Overlap between state and a stored pattern."""
        return (state @ self.patterns[pattern_idx]) / self.N


def test_retrieval(network, pattern_idx, noise_level=0.1):
    """Corrupt a pattern and try to retrieve it."""
    pattern = network.patterns[pattern_idx].copy()

    # Flip some bits
    n_flip = int(noise_level * network.N)
    flip_idx = np.random.choice(network.N, n_flip, replace=False)
    pattern[flip_idx] *= -1

    # Retrieve
    retrieved = network.update(pattern)
    overlap = network.overlap(retrieved, pattern_idx)

    return retrieved, overlap


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def update_with_history(network, state, n_steps=100):
    """Yields states after each spin flip."""
    state = state.copy()
    yield state.copy()
    for _ in range(n_steps):
        old_state = state.copy()
        for i in np.random.permutation(network.N):
            h = network.W[i] @ state
            state[i] = np.sign(h) if h != 0 else state[i]
            yield state.copy()  # moved inside the inner loop
        if np.array_equal(state, old_state):
            break


def animate_retrieval_random(network, size=20, interval=50):
    """Start from random noise and show convergence alongside all memories."""
    p = network.p

    # Random initial state
    state = np.random.choice([-1, 1], size=network.N)

    # Collect history
    history = list(update_with_history(network, state))

    # Set up figure: all memories on top row, animation below
    fig, axes = plt.subplots(2, p, figsize=(3 * p, 6))

    # Top row: stored memories
    for i in range(p):
        axes[0, i].imshow(network.patterns[i].reshape(size, size), cmap='gray', vmin=-1, vmax=1)
        axes[0, i].set_title(f'Memory {i}')
        axes[0, i].axis('off')

    # Bottom row: animation in the middle panel, hide others
    mid = p // 2
    for i in range(p):
        if i != mid:
            axes[1, i].axis('off')

    im = axes[1, mid].imshow(history[0].reshape(size, size), cmap='gray', vmin=-1, vmax=1)
    axes[1, mid].set_title('Step 0')
    axes[1, mid].axis('off')

    def update(frame):
        im.set_array(history[frame].reshape(size, size))
        # Compute overlaps with all memories
        overlaps = [network.overlap(history[frame], i) for i in range(p)]
        best = np.argmax(overlaps)
        axes[1, mid].set_title(f'Step {frame} (closest: {best}, overlap: {overlaps[best]:.2f})')
        return [im]

    anim = FuncAnimation(fig, update, frames=len(history), interval=interval, blit=True)
    plt.tight_layout()
    plt.show()
    return anim

if __name__ == '__main__':
    from patterns import generate_patterns

    patterns_2d = generate_patterns(3, size=20, correlation_length=2)
    patterns_flat = patterns_2d.reshape(3, -1)
    net = HopfieldNetwork(patterns_flat)

    animate_retrieval_random(net, interval=10)