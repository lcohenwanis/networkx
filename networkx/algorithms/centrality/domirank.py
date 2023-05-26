"""DomiRank centrality."""

import networkx as nx


def domirank(G: nx.Graph, sigma: float, theta: float = 1) -> dict:
    """Returns the DomiRank centrality of each of the nodes in the graph.

    DomiRank quantifies the dominance of the networks’
    nodes in their respective neighborhoods.

    Parameters:
    ----------
     G : graph
      A NetworkX graph.

    sigma: float
        competition factor bounded between (0, -1/λ_N),
        where λ_N represents the minimum eigenvector of A.

    theta: float, optional
        optional weight parameter to scale the domiranks (from original equation in reference paper).

    Returns
    -------
    domirank : dictionary
       Dictionary of nodes with DomiRank as value.

    Examples
    --------
    >>> G = nx.erdos_renyi_graph(n=5, p=0.5)
    >>> sigma = 0.5
    >>> domi = nx.domirank(A, sigma)


     References
    ----------
    .. [1] M. Engsig, A. Tejedor, Y. Moreno, E. Foufoula-Georgiou, C. Kasmi,
       "DomiRank Centrality: revealing structural fragility of complex networks via node dominance."
       https://arxiv.org/abs/2305.09589
    """
    import numpy as np
    import scipy as sp

    # Get number of nodes
    N = len(G)
    if N == 0:
        return {}

    # use nodelist to get Adjacency matrix
    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, dtype=float)

    # Verify the given sigma value is viable
    try:
        # compute λ_N - smallest eigenvalue of A
        e = np.linalg.eigvals(A)
        lambda_N = min(e)

        # check if sigma is between (0, -1/λ_N)
        assert sigma > 0
        assert sigma < -1 / lambda_N

    except Exception as e:
        print(f"Sigma is out of bounds. Needs to be between 0 and {-1 / lambda_N}")

    # TODO: implement domirank centrality code
    # TODO: and convert output to dictionary with key: node, value: centrality
    dr = (
        theta
        * sigma
        * np.matmul(
            np.linalg.inv(np.matmul(sigma, A) + np.identity(N)),
            np.multiply(A, np.ones((N, 1))),
        )
    )

    return dr
