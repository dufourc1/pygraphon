import numpy as np
class AssociativeFullyRandom:
    def __init__(
        self,
        K: int,
        min_inside_classes: float,
        max_inside_classes: float,
        min_outside_classes: float,
        max_outside_classes: float,
    ) -> None:
        self.Theta = np.random.uniform(min_outside_classes, max_outside_classes, (K, K))
        # make it symmetric
        self.K = K
        self.Theta = (self.Theta + self.Theta.T) / 2
        # replace diagonal with intra-cluster probabilities
        self.Theta[np.diag_indices_from(self.Theta)] = np.random.uniform(min_inside_classes, max_inside_classes, (K,))

        self.density = np.sum(self.Theta) / (self.Theta.shape[0] ** 2)

def generate_hierarchical_theta(k, m, theta_min_intra, theta_max_intra, theta_min_meta_intra, theta_max_meta_intra, theta_min_inter, theta_max_inter):
    """
    Generate a symmetric hierarchical connectivity matrix θ for an HSBM.

    Parameters:
    - k: total number of blocks
    - m: number of meta-blocks
    - theta_min_intra, theta_max_intra: intra-connectivity range within a block
    - theta_min_meta_intra, theta_max_meta_intra: intra-connectivity range within a meta-block but across blocks
    - theta_min_inter, theta_max_inter: inter-connectivity range between meta-blocks

    Returns:
    - θ: k x k symmetric hierarchical connectivity matrix
    """

    # Check if k is divisible by m
    if k % m != 0:
        raise ValueError("k should be divisible by m for a uniform hierarchical structure.")

    n = k // m  # Number of blocks within each meta-block

    theta = np.zeros((k, k))

    for i in range(k):
        for j in range(i, k):  # Only loop over the upper triangle including the diagonal
            # Determine meta-blocks for i and j
            meta_block_i = i // n
            meta_block_j = j // n

            if meta_block_i == meta_block_j:
                # Within the same meta-block
                if i == j:
                    # Same block: intra-connectivity
                    theta[i, j] = np.random.uniform(theta_min_intra, theta_max_intra)
                else:
                    # Different block but same meta-block: meta-intra-connectivity
                    theta[i, j] = np.random.uniform(theta_min_meta_intra, theta_max_meta_intra)
            else:
                # Different meta-blocks: inter-connectivity
                theta[i, j] = np.random.uniform(theta_min_inter, theta_max_inter)
            
            # Make the matrix symmetric
            theta[j, i] = theta[i, j]

    return theta