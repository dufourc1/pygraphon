from typing import Tuple
import numpy as np
from pygraphon.graphons import Graphon


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
        self.Theta[np.diag_indices_from(self.Theta)] = np.random.uniform(
            min_inside_classes, max_inside_classes, (K,)
        )

        self.density = np.sum(self.Theta) / (self.Theta.shape[0] ** 2)

class AssociativeSimpleNonDiagonal:
    def __init__(
        self,
        K: int,
        min_inside_classes: float,
        max_inside_classes: float,
        min_outside_classes: float,
        max_outside_classes: float,
    ) -> None:

        self.Theta = np.ones(K, K) * np.random.uniform(
            min_outside_classes, max_outside_classes
        )
        self.K = K
        # replace diagonal with intra-cluster probabilities
        self.Theta[np.diag_indices_from(self.Theta)] = np.random.uniform(
            min_inside_classes, max_inside_classes, (K,)
        )

        self.density = np.sum(self.Theta) / (self.Theta.shape[0] ** 2)
        
class ER:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
        self.Theta = np.array([self.p]).reshape(1, 1)
        self.K = 1

def gen_sbm(n_list: list,
            k_list: list,
            theta_intra_min: float,
            theta_intra_max: float,
            theta_inter_min: float,
            theta_inter_max: float,
            folder_name: str) -> Tuple[Graphon, np.ndarray]:
    """Generates the csv file to store results when testing estimator for different graphons.

    Parameters
    ----------
    n_list : list
        list of n
    k_list : list
        list of k
    theta_intra_min : float
        minimum value of theta_intra
    theta_intra_max : float
        maximum value of theta_intra
    theta_inter_min : float
        minimum value of theta_inter
    theta_inter_max : float
        maximum value of theta_inter
    folder_name : str
        folder to save the results
    """
    