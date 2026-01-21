import numpy as np
from hopfield.network import generate_network


def measure_cross_overlap(N, xi, M=200, num_networks=200):
    """Measure average cross-overlap for given N and xi."""
    n_neurons = N * N
    all_cross = []

    for _ in range(num_networks):
        patterns, _ = generate_network(N, M, xi)
        cross = np.abs(patterns @ patterns.T) / n_neurons
        np.fill_diagonal(cross, 0)
        all_cross.append(cross.sum() / (M * (M - 1)))

    return np.mean(all_cross), np.std(all_cross)


if __name__ == '__main__':
    N = 40
    xi_values = [.01, 0.1, 0.2,0.3,0.4, 0.5,0.6, 0.7,0.8,0.9, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,20]

    print(f"N = {N}")
    print("xi\t\t\t\txi^2\t\t\t\ttcross")
    for xi in xi_values:
        mean, std = measure_cross_overlap(N, xi, M=200, num_networks=200)
        print(f"{xi:1f}\t\t{xi**2:1f}\t\t{N*(mean-0.79768/N):.5f}")