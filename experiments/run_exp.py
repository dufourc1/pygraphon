import io
import logging
import pickle

import numpy as np
from loguru import logger
from tqdm import tqdm
import os
import shutil
from pygraphon.graphons import Graphon
from pygraphon import estimators
from pygraphon.estimators.SAS import SAS
from pygraphon.estimators.UniversalSingularValueDecomposition import UVST
from pygraphon.estimators.sba import SBA
from .utils_data import parser_argument, clean_temp_folder, csv_gen_graphon
from .utils_generators import AssociativeFullyRandom, AssociativeSimpleNonDiagonal, ER


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)


def main(adjacency_matrix: np.ndarray,
         graphon: Graphon,
         method: str,
         folder_name: str,
         ):

    logger.info("Starting experiment")
    # Initialise experiments
    # ----------------------
    csv_fill = {"Type of Graphon": None, "n": None, "MSE": None, "MAE": None, "AUC": None, "AUPRC": None}
    method_est = getattr(estimators, method)
    # Run experiments
    # ----------------------
    ## Running different estimators for comparison
    est1 = SAS().fit(adjacency_matrix)
    est2 = UVST().fit(adjacency_matrix)
    est3 = SBA().fit(adjacency_matrix)

    est = method_est().fit(adjacency_matrix)
    ## Running SmoothNetHist

    # ----------------------
    logger.info("Computing done")
    with open("results"+f"{folder_name}.pkl", "wb") as f:
        pickle.dump(metric_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Cleaning {folder_name}/")
    clean_temp_folder(folder_name)

    logger.info("Results saved, bye")


if __name__ == "__main__":
    # init logger because Charles corrupted me
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    logger.add("log_exp.txt", level="INFO", mode="w")

    # sparse regime import and init
    folder_name = "sbm_ssm"
    mult_rep = True
    parser_argument(folder_name)

    n = 200

    # If generate with SBM-SSM
    if folder_name == "sbm_ssm":
        sbm_params = {}
        theta = AssociativeFullyRandom()
        k_max = 5
        # theta init
        theta_intra_min, theta_intra_max = 0.15, 0.5
        theta_inter_min, theta_inter_max = 0.05, 0.15
        if mult_rep:
            # n init
            n_min, n_step = 50, 50
            n_list = np.arange(n_min, n, n_step)
            # k init
            k_list = list(range(2, k_max + 1))
            graphon = []
            adjacency_matrix = []
            for i,k in enumerate(k_list):
                graphon[i] = theta.Theta(K=k, min_inside_classes=theta_intra_min, max_inside_classes=theta_intra_max, min_outside_classes=theta_inter_min, max_outside_classes=theta_inter_max)
                adjacency_matrix[i] = graphon[i].sample(n) #not good as doesnt take into account the number of nodes in each class
        else:
            graphon = theta.Theta(K=k_max, min_inside_classes=theta_intra_min, max_inside_classes=theta_intra_max, min_outside_classes=theta_inter_min, max_outside_classes=theta_inter_max)
            adjacency_matrix = graphon.sample(n)
    method = ["SmoothNetHist", "SAS", "UVST", "SBA"]
    for m in method:
        main(adjacency_matrix= A, graphon=graphon, method=m, folder_name=folder_name)
