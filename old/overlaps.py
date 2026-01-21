import numpy as np


def test_capacity(M, N, xi, c, noise_level=0.2):
    """
    Test retrieval on c*M corrupted images.

    Returns:
        overlaps: array of length c*M, overlap with target after convergence
        cross_overlap: average |overlap| between distinct memories
    """
    from hopfield.network import generate_network

    patterns, W = generate_network(N, M, xi)
    n_neurons = N * N

    # Cross-overlaps between memories
    cross = np.abs(patterns @ patterns.T) / n_neurons
    np.fill_diagonal(cross, 0)  # exclude self-overlap
    cross_overlap = cross.sum() / (M * (M - 1))  # average over off-diagonal

    overlaps = []

    for trial in range(c):
        for idx in range(M):
            # Corrupt pattern idx
            state = patterns[idx].copy()
            n_flip = int(noise_level * n_neurons)
            flip_idx = np.random.choice(n_neurons, n_flip, replace=False)
            state[flip_idx] *= -1

            # Run dynamics
            for _ in range(100):
                old_state = state.copy()
                for i in np.random.permutation(n_neurons):
                    h = W[i] @ state
                    state[i] = np.sign(h) if h != 0 else state[i]
                if np.array_equal(state, old_state):
                    break

            # Measure overlap with target
            overlap = np.abs(state @ patterns[idx]) / n_neurons
            overlaps.append(overlap)

    return np.array(overlaps), cross_overlap

if __name__ == '__main__':
    for xi in [0.5, 1.0, 2.0, 4.0]:
        overlaps, cross = test_capacity(M=20, N=20, xi=xi, c=5)
        print(f"xi={xi}: retrieval={np.mean(overlaps):.3f}, cross-overlap={cross:.3f}")