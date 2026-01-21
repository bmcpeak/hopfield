import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize_retrieval(patterns, W, N, interval=10, noise_level=0.5):
    """
    Two animations: one from random, one from perturbed memory.
    """
    M = len(patterns)
    n_neurons = N * N

    def run_dynamics(initial_state):
        state = initial_state.copy()
        history = [state.copy()]
        for _ in range(100):
            old_state = state.copy()
            for i in np.random.permutation(n_neurons):
                h = W[i] @ state
                state[i] = np.sign(h) if h != 0 else state[i]
                history.append(state.copy())
            if np.array_equal(state, old_state):
                break
        return history

    # Random start
    random_state = np.random.choice([-1, 1], size=n_neurons)
    history_random = run_dynamics(random_state)

    # Perturbed memory
    which_memory = np.random.randint(M)
    perturbed_state = patterns[which_memory].copy()
    n_flip = int(noise_level * n_neurons)
    flip_idx = np.random.choice(n_neurons, n_flip, replace=False)
    perturbed_state[flip_idx] *= -1
    history_perturbed = run_dynamics(perturbed_state)

    # Pad to same length
    max_len = max(len(history_random), len(history_perturbed))
    history_random += [history_random[-1]] * (max_len - len(history_random))
    history_perturbed += [history_perturbed[-1]] * (max_len - len(history_perturbed))

    # Set up figure
    fig, axes = plt.subplots(2, M, figsize=(2.5 * M, 5))
    if M == 1:
        axes = axes.reshape(2, 1)

    # Top row: stored memories
    for i in range(M):
        axes[0, i].imshow(patterns[i].reshape(N, N), cmap='gray', vmin=-1, vmax=1)
        axes[0, i].set_title(f'Memory {i}')
        axes[0, i].axis('off')

    # Bottom row: two animations
    left = 1
    right = M - 2
    for i in range(M):
        if i not in [left, right]:
            axes[1, i].axis('off')

    im_random = axes[1, left].imshow(history_random[0].reshape(N, N), cmap='gray', vmin=-1, vmax=1)
    axes[1, left].axis('off')
    title_random = axes[1, left].set_title('Random: Step 0')

    im_perturbed = axes[1, right].imshow(history_perturbed[0].reshape(N, N), cmap='gray', vmin=-1, vmax=1)
    axes[1, right].axis('off')
    title_perturbed = axes[1, right].set_title(f'Perturbed {which_memory}: Step 0')

    def overlap(state, pattern):
        return (state @ pattern) / len(state)

    def update(frame):
        im_random.set_array(history_random[frame].reshape(N, N))
        overlaps_r = [overlap(history_random[frame], patterns[i]) for i in range(M)]
        overlap_str_r = ', '.join([f'{ov:.2f}' for ov in overlaps_r])
        title_random.set_text(f'Random: [{overlap_str_r}]')

        im_perturbed.set_array(history_perturbed[frame].reshape(N, N))
        overlaps_p = [overlap(history_perturbed[frame], patterns[i]) for i in range(M)]
        overlap_str_p = ', '.join([f'{ov:.2f}' for ov in overlaps_p])
        title_perturbed.set_text(f'Perturbed {which_memory}: [{overlap_str_p}]')

        return [im_random, im_perturbed, title_random, title_perturbed]

    anim = FuncAnimation(fig, update, frames=max_len, interval=interval, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()
    return anim


if __name__ == '__main__':
    from hopfield.network import generate_network

    patterns, W = generate_network(N=30, M=5, xi=2.0)
    visualize_retrieval(patterns, W, N=30, interval=5)