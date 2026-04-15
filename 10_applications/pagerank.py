import numpy as np


def build_transition_matrix(adjacency):
    """
    Build a column-stochastic transition matrix from an adjacency matrix.
    """
    adjacency = np.asarray(adjacency, dtype=np.float64)
    n = adjacency.shape[0]

    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square.")

    transition = adjacency.copy()
    out_degree = np.sum(transition, axis=0)

    for j in range(n):
        if out_degree[j] == 0:
            transition[:, j] = 1.0 / n
        else:
            transition[:, j] /= out_degree[j]

    return transition


def pagerank(adjacency, damping=0.85, tol=1e-10, max_iterations=100):
    """
    Compute PageRank scores by power iteration on the Google matrix.
    """
    transition = build_transition_matrix(adjacency)
    n = transition.shape[0]

    google = damping * transition + (1.0 - damping) * np.ones((n, n)) / n
    rank = np.ones(n, dtype=np.float64) / n

    for iteration in range(1, max_iterations + 1):
        rank_next = google @ rank
        if np.linalg.norm(rank_next - rank, ord=1) < tol:
            return rank_next / np.sum(rank_next), {"iterations": iteration, "converged": True}
        rank = rank_next

    return rank / np.sum(rank), {"iterations": max_iterations, "converged": False}


if __name__ == "__main__":
    # Example graph: 0 points to 1 and 2, 1 points to 2, 2 points to 0
    adj = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [1, 1, 0]
    ])
    
    ranks, info = pagerank(adj)
    print("--- PageRank Example ---")
    print(f"Adjacency matrix:\n{adj}")
    print(f"Converged: {info['converged']} in {info['iterations']} iterations")
    print(f"Ranks: {ranks}")
    print(f"Rank sum: {np.sum(ranks)}")
