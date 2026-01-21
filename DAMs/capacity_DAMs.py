import numpy as np
import matplotlib.pyplot as plt
import os
import random


def generate_patterns(N, M, xi):
    """Generate M patterns of size N x N with correlation length xi."""
    from patterns import generate_patterns as gen_pat
    return gen_pat(M, size=N, correlation_length=xi).reshape(M, -1)


def softmax(z, beta):
    z = z - z.max()
    exp_z = np.exp(beta * z)
    return exp_z / exp_z.sum()


def test_capacity(M, N, xi, beta=5.0, num_networks=10, trials_per_pattern=5, noise_level=0.0):
    """
    Test retrieval across multiple independent networks.
    """
    n_neurons = N * N
    all_overlaps = []

    for _ in range(num_networks):
        patterns = generate_patterns(N, M, xi)

        for trial in range(trials_per_pattern):
            for idx in range(min(M,100)):
                state = patterns[idx].copy().astype(float)
                n_flip = int(noise_level * n_neurons)
                flip_idx = np.random.choice(n_neurons, n_flip, replace=False)
                state[flip_idx] *= -1

                for _ in range(100):
                    old_state = state.copy()
                    for i in np.random.permutation(n_neurons):
                        similarities = patterns @ state
                        weights = softmax(similarities, beta)
                        state[i] = patterns[:, i] @ weights
                    if np.allclose(state, old_state):
                        break

                overlap = np.abs(state @ patterns[idx]) / n_neurons
                all_overlaps.append(overlap)

    return np.mean(all_overlaps)


def measure_cross_overlap(M, N, xi, num_networks=10):
    """Measure average cross-overlap for given parameters."""
    n_neurons = N * N
    all_cross = []

    for _ in range(num_networks):
        patterns = generate_patterns(N, M, xi)
        cross = np.abs(patterns @ patterns.T) / n_neurons
        np.fill_diagonal(cross, 0)
        all_cross.append(cross.sum() / (M * (M - 1)))

    return np.mean(all_cross)


def capacity_plot(xi, N=20, M_max=60, beta=5.0, num_networks=10, trials_per_pattern=3, jump=1, noise_level=0.2):
    """
    Plot retrieval overlap vs M for a given xi and beta.
    """
    M_values = list(range(jump, M_max + 1, jump))
    retrievals = []

    for M in M_values:
        retrieval = test_capacity(M, N, xi, beta, num_networks, trials_per_pattern, noise_level)
        retrievals.append(retrieval)
        print(f"M={M}: retrieval={retrieval:.3f}")

    # Measure cross-overlap only at M_max
    avg_cross = measure_cross_overlap(100, N, xi, 100)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.set_xlabel('Number of memories (M)')
    ax.set_ylabel('Retrieval overlap')
    ax.plot(M_values, retrievals, 'o-', color='blue')
    ax.set_ylim(0, 1.05)

    plt.title(f'DAM: N={N}², ξ={xi}, β={beta}, cross≈{avg_cross:.3f}, nets={num_networks}, trials={trials_per_pattern}, noise={noise_level}')
    fig.tight_layout()

    # Save to runs folder
    os.makedirs('runs', exist_ok=True)
    run_id = random.randint(1000, 9999)
    filename = f'runs/dam_capacity_N{N}_xi{xi}_beta{beta}_{run_id}.png'
    plt.savefig(filename, dpi=150)
    print(f'Saved to {filename}')

    plt.show(block=False)

    return M_values, retrievals, avg_cross


if __name__ == '__main__':
    xi = 0.1
    capacity_plot(xi, N=8, M_max=200000, beta=3.0, num_networks=5, trials_per_pattern=2, jump=10000, noise_level=0.2)

    xi = 2
    capacity_plot(xi, N=8, M_max=20000, beta=3.0, num_networks=5, trials_per_pattern=2, jump=200, noise_level=0.2)