import importlib.util
from typing import Tuple
import numpy as np
from pygraphon.graphons import Graphon


def class_extract(class_name: str,
                  file_path: str
                  ) -> type:
    """Extracts a class from a file.

    Parameters
    ----------
    class_name : str
        name of the class to extract
    file_path : str
        path to the file containing the class. Example: "src\pygraphon\graphons\StepGraphon.py"
    
    Returns
    -------
    type
        class object
    
    Raises
    ------
    TypeError
        if the retrieved object is not a class
    NameError
        if the class is not found in the file
    """
    try:
        # Load the module from the file path
        spec = importlib.util.spec_from_file_location("module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Check if the class exists in the module
        if hasattr(module, class_name):
            # Retrieve the class object using the class name
            class_object = getattr(module, class_name)

            # Check if the retrieved object is a class
            if isinstance(class_object, type):
                return class_object
            else:
                raise TypeError(f"{class_name} is not a class.")
        else:
            raise NameError(f"Class {class_name} not found in {file_path}.")
    except Exception as e:
        print(f"Error: {e}")

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
    