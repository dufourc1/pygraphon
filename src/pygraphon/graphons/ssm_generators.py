import numpy as np
from numpy.random import uniform


class AssociativeFullyRandom:
    """Generate a random theta matrix for a stochastic block model."""

    def __init__(
        self,
        K: int,
        min_inside_classes: float,
        max_inside_classes: float,
        min_outside_classes: float,
        max_outside_classes: float,
    ) -> None:
        """Generate a random theta matrix for a stochastic block model.

        Parameters
        ----------
        K : int
            Number of classes.
        min_inside_classes : float
            Minimum intra-connectivity within a class.
        max_inside_classes : float
            Maximum intra-connectivity within a class.
        min_outside_classes : float
            Minimum inter-connectivity between classes.
        max_outside_classes : float
            Maximum inter-connectivity between classes.
        """
        self.Theta = uniform(min_outside_classes, max_outside_classes, (K, K))
        # make it symmetric
        self.K = K
        self.Theta = (self.Theta + self.Theta.T) / 2
        # replace diagonal with intra-cluster probabilities
        self.Theta[np.diag_indices_from(self.Theta)] = uniform(
            min_inside_classes, max_inside_classes, (K,)
        )

        self.density = np.sum(self.Theta) / (self.Theta.shape[0] ** 2)


def generate_hierarchical_theta(
    k,
    m,
    theta_min_intra,
    theta_max_intra,
    theta_min_meta_intra,
    theta_max_meta_intra,
    theta_min_inter,
    theta_max_inter,
):
    """
    Generate a hierarchical theta matrix for a hierarchical stochastic block model.

    Parameters
    ----------
    k : int
        Number of nodes in the graph.
    m : int
        Number of nodes in each meta-block.
    theta_min_intra : float
        Minimum intra-connectivity within a block.
    theta_max_intra : float
        Maximum intra-connectivity within a block.
    theta_min_meta_intra : float
        Minimum meta-intra-connectivity within a meta-block.
    theta_max_meta_intra : float
        Maximum meta-intra-connectivity within a meta-block.
    theta_min_inter : float
        Minimum inter-connectivity between meta-blocks.
    theta_max_inter : float
        Maximum inter-connectivity between meta-blocks.

    Returns
    -------
    np.ndarray
        A k-by-k matrix of theta values.

    Raises
    ------
    ValueError
        k should be divisible by m for a uniform hierarchical structure.

    Examples
    --------
    >>> theta = generate_hierarchical_theta(k=9,
                                            m=3,
                                            theta_min_intra=0.5,
                                            theta_max_intra=0.7,
                                            theta_min_meta_intra=0.4,
                                            theta_max_meta_intra=0.4,
                                            theta_min_inter=0.1,
                                            theta_max_inter=0.3)
    >>> graphon_hsbm = StepGraphon(theta, bandwidthHist=1 / k)
    >>> graphon_hsbm.repr = "hsbm"
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
                    theta[i, j] = uniform(theta_min_intra, theta_max_intra)
                else:
                    # Different block but same meta-block: meta-intra-connectivity
                    theta[i, j] = uniform(theta_min_meta_intra, theta_max_meta_intra)
            else:
                # Different meta-blocks: inter-connectivity
                theta[i, j] = uniform(theta_min_inter, theta_max_inter)

            # Make the matrix symmetric
            theta[j, i] = theta[i, j]

    return theta
