import io
import logging
import pickle

import numpy as np
from loguru import logger
from tqdm import tqdm

# import os
# import shutil
from pygraphon.graphons import Graphon
from pygraphon.graphons import StepGraphon
from pygraphon.graphons import common_graphons
from pygraphon import estimators
from pygraphon.estimators import USVT, Completion, SAS, SmoothNetHist
from utils_data import (
    parser_argument,
    clean_temp_folder,
    csv_gen_graphon,
    csv_store_StepGraphon_parameters,
    update_row,
)
from utils_generators import AssociativeFullyRandom, AssociativeSimpleNonDiagonal, ER
from pygraphon.norm import MseProbaEdge, MaeProbaEdge, AUCEdge, AUPRCEdge
from typing import List, Optional
from scipy.special import comb


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


def main(
    adjacency_matrix: np.ndarray,
    graphon: Optional[Graphon],
    method: str,
    folder_name: str,
):
    logger.info("Starting experiment")
    # Initialise experiments
    # ----------------------
    method_est = getattr(estimators, method)
    # Run experiments
    # ----------------------
    est = method_est()
    est.fit(adjacency_matrix)
    n = adjacency_matrix.shape[0]
    if graphon is not None:
        norm_mse = MseProbaEdge(n_nodes=n)(graphon, est.get_graphon())
        norm_mae = MaeProbaEdge(n_nodes=n)(graphon, est.get_graphon())
    else:
        norm_mse = MseProbaEdge(n_nodes=n)(adjacency_matrix, est.get_graphon())
        norm_mae = MaeProbaEdge(n_nodes=n)(adjacency_matrix, est.get_graphon())
    # Compute AUC and AUPRC
    norm_auc = AUCEdge(n_nodes=n)(adjacency_matrix, est.get_edge_connectivity())
    norm_auprc = AUPRCEdge(n_nodes=n)(adjacency_matrix, est.get_edge_connectivity())
    # if SmoothNetHist, compute BIC and ratio_par, leave None otherwise
    if method == "SmoothNetHist":
        graphon_repr = (
            f"sbm {len(np.unique(est.get_graphon().graphon))}" if isinstance(graphon, StepGraphon) else graphon.repr
        )
    if method == "HistogramEstimator":
        graphon_repr = (
            f"sbm {comb(est.get_graphon().graphon.shape[0]+1,2)}" if isinstance(graphon, StepGraphon) else graphon.repr
        )
    if method == "SmoothNetHist":
        result = [
            method,
            graphon_repr,
            n,
            norm_mse,
            norm_mae,
            norm_auc,
            norm_auprc,
            est.get_bic(),
            est.get_ratio_par(),
        ]
    else:
        result = [method, graphon_repr, n, norm_mse, norm_mae, norm_auc, norm_auprc, None, None]
    # ----------------------
    logger.info("Computing done")
    with open("results" + f"{folder_name}.pkl", "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Cleaning {folder_name}/")
    # clean_temp_folder(folder_name)

    logger.info("Results saved, bye")

    return result


if __name__ == "__main__":
    # init logger because Charles corrupted me
    # logger.remove()
    # logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    # logger.add("log_exp.txt", level="INFO", mode="w")

    # sparse regime import and init
    folder_name = "usual_graphon"
    mult_rep = True
    # parser_argument(folder_name)

    n = 100
    n += 1

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # If generate with SBM-SSM
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    if folder_name == "sbm_ssm":
        k_max = 2
        # theta init
        theta_intra_min, theta_intra_max = 0.5, 0.7
        theta_inter_min, theta_inter_max = 0.1, 0.4

        # -----------------------------------
        # ----- If multiple repetitions -----
        # -----------------------------------
        if mult_rep:
            # n init
            n_min, n_step = 50, 50
            n_list = np.arange(n_min, n, n_step)
            # k init
            k_list = list(range(2, k_max + 1))
            csv_store_StepGraphon_parameters(
                path="results/sbm_ssm",
                file_name="parameters" + f"_{folder_name}" + ".csv",
                n_it=len(n_list) * len(k_list),
                SBM_SSM=True,
            )
            csv_gen_graphon(
                path="results/sbm_ssm", file_name=f"_{folder_name}" + ".csv", n_graphons=len(n_list) * len(k_list)
            )
            for i, k in enumerate(k_list):
                theta = AssociativeFullyRandom(
                    K=k,
                    min_inside_classes=theta_intra_min,
                    max_inside_classes=theta_intra_max,
                    min_outside_classes=theta_inter_min,
                    max_outside_classes=theta_inter_max,
                )
                update_row(
                    folder_name="sbm_ssm",
                    file_name="parameters" + f"_{folder_name}",
                    new_row=theta.Theta,
                    row_index=i + 1,
                )
                graphon = StepGraphon(
                    theta.Theta, bandwidthHist=1 / k
                )  # can add arg repr to specify the name as a str (default is "Step Graphon")
                for j, n in enumerate(n_list):
                    result = main(
                        adjacency_matrix=graphon.draw(n=n, exchangeable=False),
                        graphon=graphon,
                        method="SmoothNetHist",
                        folder_name=folder_name,
                    )
                    update_row(
                        folder_name="sbm_ssm",
                        file_name=f"_{folder_name}",
                        new_row=result,
                        row_index=i * len(n_list) + j + 1,
                    )
        # ------------------------------------
        # ------- If single repetition -------
        # ------------------------------------
        else:
            csv_gen_graphon(
                path="results/sbm_ssm", file_name="comparison" + f"_{folder_name}" + ".csv", n_graphons=k_max
            )
            theta = AssociativeFullyRandom(
                K=k_max,
                min_inside_classes=theta_intra_min,
                max_inside_classes=theta_intra_max,
                min_outside_classes=theta_inter_min,
                max_outside_classes=theta_inter_max,
            )
            graphon = StepGraphon(theta.Theta, bandwidthHist=1 / k_max)
            result = main(
                adjacency_matrix=graphon.draw(n=n, exchangeable=False),
                graphon=graphon,
                method="SmoothNetHist",
                folder_name=folder_name,
            )
            update_row(
                folder_name="sbm_ssm",
                file_name="comparison" + f"_{folder_name}",
                new_row=result,
                row_index=1,
            )

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # If generate with other graphons
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    graphons = []
    if folder_name == "usual_graphon":
        method = ["SAS", "USVT", "Completion"]#"SmoothNetHist", "HistogramEstimator", 
        for g in common_graphons.values():
            graphons.append(g)
        csv_gen_graphon(
                    path="results/usual_graphon",
                    file_name="method_comp" + f"_{folder_name}" + ".csv",
                    n_graphons=len(graphons) * len(method),
                )

        for i, m in enumerate(method):
            print(m)
            for j, g in enumerate(graphons):
                result = main(
                    adjacency_matrix=g.draw(n=n, exchangeable=False),
                    graphon=g,
                    method=m,
                    folder_name=folder_name,
                )
                update_row(
                    folder_name="usual_graphon",
                    file_name="method_comp" + f"_{folder_name}",
                    new_row=result,
                    row_index=i * len(graphons) + j + 1,
                )
