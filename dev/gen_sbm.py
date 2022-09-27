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