import numpy as np

def generate_wpe(N, alpha, seed=None):
    if seed is not None:
        np.random.seed(seed)

    P = int(np.floor(alpha * N))
    if P == 0:
        return np.zeros((N, N)), 0.0

    W_rand = np.random.randn(N, P)
    col_means = W_rand.mean(axis=0)
    W_alpha = W_rand - col_means

    J_tilde = (-1 / N) * (W_alpha @ W_alpha.T)
    J = J_tilde - np.diag(np.diag(J_tilde))
    min_energy = -0.5 * np.sum(J)

    return J, min_energy
