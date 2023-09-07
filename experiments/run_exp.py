import pickle

import numpy as np
from loguru import logger
import sys
import os
# import shutil
from pygraphon.graphons import Graphon
from pygraphon.graphons import StepGraphon
from pygraphon.graphons import common_graphons
from pygraphon import estimators
from utils_data import (
    csv_gen_graphon,
    csv_store_StepGraphon_parameters,
    update_row,
    get_result_with_std,
)
from utils_generators import AssociativeFullyRandom
from pygraphon.norm import MseProbaEdge, MaeProbaEdge, AUCEdge, AUPRCEdge
from typing import Optional
from scipy.special import comb

CRITERIONS = ["bic", "aic", "elbow", "mallows_cp", "hqic", "fpe"]


def setup_logger_file(folder_name: str):
    logger.remove()
    logger.enable("pygraphon")
    logger.add(os.path.join(folder_name, "log_exp.txt"), level="TRACE", mode="w",
               format="{time:YYYY-MM-DD at HH:mm:ss} | {file} | line {line} | {level} | {message}")
    logger.add(os.path.join(folder_name, "log_errors.txt"), level="CRITICAL", mode="w",
               format="{time:YYYY-MM-DD at HH:mm:ss} | {file} | line {line} | | {level} | {message}")
    logger.add(sys.stderr, level="INFO",  format="{time:YYYY-MM-DD at HH:mm:ss} | {file} | line {line} | {level} | {message}")
    logger.info("Logger set up")


def main(
    adjacency_matrix: np.ndarray,
    graphon: Optional[Graphon],
    method: str,
    folder_name: str,
):
    # Initialise experiments
    # ----------------------
    method_est = getattr(estimators, method)
    # Run experiments
    # ----------------------
    if method == "SmoothNetHist":
        est = method_est("all")
    else:
        est = method_est()
    n = adjacency_matrix.shape[0]
    try:
        if method == "SmoothNetHist":
            result = []
            est.fit(adjacency_matrix)
            est.get_ratio_par()
            for criterion in CRITERIONS:
                if graphon is not None:
                    norm_mse = MseProbaEdge(n_nodes=n)(
                        graphon, est.best_pairs[criterion]["graphon"])
                    norm_mae = MaeProbaEdge(n_nodes=n)(
                        graphon, est.best_pairs[criterion]["graphon"])
                else:
                    norm_mse = MseProbaEdge(n_nodes=n)(
                        adjacency_matrix, est.best_pairs[criterion]["graphon"])
                    norm_mae = MaeProbaEdge(n_nodes=n)(
                        adjacency_matrix, est.best_pairs[criterion]["graphon"])
                # Compute AUC and AUPRC
                norm_auc = AUCEdge(n_nodes=n)(
                    adjacency_matrix, est.best_pairs[criterion]["pij"])
                norm_auprc = AUPRCEdge(n_nodes=n)(
                    adjacency_matrix, est.best_pairs[criterion]["pij"])
                # if SmoothNetHist, compute BIC and ratio_par, leave None otherwise
                s = len(
                    np.unique(est.best_pairs[criterion]["graphon"].graphon))
                graphon_repr = (
                    f"ssm {s}"
                    if isinstance(graphon, StepGraphon)
                    else graphon.repr
                )
                sub_result = [method, graphon_repr, n, norm_mse, norm_mae, norm_auc,
                              norm_auprc, est.criterion_values[criterion], est._num_par_smooth[criterion]]
                result.append(sub_result)
        elif method == "HistogramEstimator":
            est.fit(adjacency_matrix)
            if graphon is not None:
                norm_mse = MseProbaEdge(n_nodes=n)(
                    graphon, est.get_graphon())
                norm_mae = MaeProbaEdge(n_nodes=n)(
                    graphon, est.get_graphon())
            else:
                norm_mse = MseProbaEdge(n_nodes=n)(
                    adjacency_matrix, est.get_graphon())
                norm_mae = MaeProbaEdge(n_nodes=n)(
                    adjacency_matrix, est.get_graphon())
            # Compute AUC and AUPRC
            norm_auc = AUCEdge(n_nodes=n)(
                adjacency_matrix, est.get_edge_connectivity())
            norm_auprc = AUPRCEdge(n_nodes=n)(
                adjacency_matrix, est.get_edge_connectivity())
            graphon_repr = (
                f"sbm {comb(est.get_graphon().graphon.shape[0]+1,2)}"
                if isinstance(graphon, StepGraphon)
                else graphon.repr
            )
            result = [method, graphon_repr, n, norm_mse,
                      norm_mae, norm_auc, norm_auprc, 0, 0]
        else:
            est.fit(adjacency_matrix)
            if graphon is not None:
                norm_mse = MseProbaEdge(n_nodes=n)(
                    graphon, est.get_graphon())
                norm_mae = MaeProbaEdge(n_nodes=n)(
                    graphon, est.get_graphon())
            else:
                norm_mse = MseProbaEdge(n_nodes=n)(
                    adjacency_matrix, est.get_graphon())
                norm_mae = MaeProbaEdge(n_nodes=n)(
                    adjacency_matrix, est.get_graphon())
            # Compute AUC and AUPRC
            norm_auc = AUCEdge(n_nodes=n)(
                adjacency_matrix, est.get_edge_connectivity())
            norm_auprc = AUPRCEdge(n_nodes=n)(
                adjacency_matrix, est.get_edge_connectivity())
            graphon_repr = graphon.repr if graphon is not None else None
            result = [method, graphon_repr, n, norm_mse,
                      norm_mae, norm_auc, norm_auprc, 0, 0]
    except Exception as e:
        logger.critical(f"Error with {method} on {graphon.repr} with n={n}")
        logger.critical(e)
        result = [method, None, n, np.nan, np.nan, np.nan, np.nan, 0, 0]
    # ----------------------
    logger.info("Saving results ")
    with open("results" + f"{folder_name}.pkl", "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    # logger.info(f"Cleaning {folder_name}/")
    # clean_temp_folder(folder_name)

    # logger.info("Results saved, bye")

    return result


if __name__ == "__main__":
    setup_logger_file(folder_name="results")
    # sparse regime import and init
    folder_name = "usual_graphon"
    mult_par = True
    mult_rep = 2
    n_min, n_max, n_step = 50, 300, 50
    n_list = np.arange(n_min, n_max+1, n_step)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # If generate with SBM-SSM
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    if folder_name == "sbm_ssm":
        k_max = 8
        # theta init
        theta_intra_min, theta_intra_max = 0.5, 0.7
        theta_inter_min, theta_inter_max = 0.1, 0.4

        # -----------------------------------
        # ----- If multiple repetitions -----
        # -----------------------------------
        if mult_par:
            # k init
            k_list = list(range(2, k_max + 1))
            csv_store_StepGraphon_parameters(
                path="results/sbm_ssm",
                file_name="parameters" + f"_{folder_name}" + ".csv",
                n_it=len(n_list) * len(k_list),
                SBM_SSM=True,
            )
            csv_gen_graphon(
                path="results/sbm_ssm",
                file_name=f"mult_{folder_name}" + ".csv",
                n_graphons=len(n_list) * len(k_list),
                smoothnethist_par=True,
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
                    result = []
                    for r in range(mult_rep):
                        result.append(main(
                            adjacency_matrix=graphon.draw(
                                n=n, exchangeable=False),
                            graphon=graphon,
                            method="SmoothNetHist",
                            folder_name=folder_name,
                        ))
                    for crit_idx, criterion in CRITERIONS:
                        update_row(
                            folder_name="sbm_ssm",
                            file_name=f"_{folder_name}",
                            new_row=get_result_with_std(
                                [sublist[crit_idx] for sublist in result]),
                            row_index=i *
                            len(n_list)*len(CRITERIONS) + j *
                            len(CRITERIONS) + crit_idx + 1,
                        )
        # ------------------------------------
        # ------- If single repetition -------
        # ------------------------------------
        else:
            csv_gen_graphon(
                path="results/sbm_ssm",
                file_name="comparison" + f"_{folder_name}" + ".csv",
                n_graphons=k_max,
                smoothnethist_par=True,
            )
            theta = AssociativeFullyRandom(
                K=k_max,
                min_inside_classes=theta_intra_min,
                max_inside_classes=theta_intra_max,
                min_outside_classes=theta_inter_min,
                max_outside_classes=theta_inter_max,
            )
            graphon = StepGraphon(theta.Theta, bandwidthHist=1 / k_max)
            result = []
            for r in range(mult_rep):
                result.append(main(
                    adjacency_matrix=graphon.draw(n=n_max, exchangeable=False),
                    graphon=graphon,
                    method="SmoothNetHist",
                    folder_name=folder_name,
                ))
            update_row(
                folder_name="sbm_ssm",
                file_name="comparison" + f"_{folder_name}",
                new_row=get_result_with_std(result),
                row_index=1,
            )

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # If generate with other graphons
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    if folder_name == "usual_graphon":
        graphons = []
        # , "USVT", "NBDsmooth", "Completion"]
        method = ["SmoothNetHist", "HistogramEstimator", "USVT", "NBDsmooth", "Completion"]
        for g in common_graphons.values():
            graphons.append(g)
        csv_gen_graphon(
            path="results/usual_graphon",
            file_name="method_comp" + f"_{folder_name}" + ".csv",
            n_graphons=len(graphons) * (len(method)+len(CRITERIONS)-1) *len(n_list)+1,
            smoothnethist_par=True,
        )
        idx = 1
        for j, g in enumerate(graphons):
            for i, m in enumerate(method):
                for k, n in enumerate(n_list):
                    logger.info(f"Graphon {g.repr} with method {m} and n={n}")
                    result = []
                    for r in range(mult_rep):
                        logger.info(f"Repetition {r+1}/{mult_rep}")
                        np.random.seed(42+k+(r+1)*(len(n_list)+1))
                        adj_mat = g.draw(n=n, exchangeable=False)
                        inter_result = main(
                            adjacency_matrix=adj_mat,
                            graphon=g,
                            method=m,
                            folder_name=folder_name,
                        )
                        result.append(inter_result)
                    if m == "SmoothNetHist":
                        for crit_idx, criterion in enumerate(CRITERIONS):
                            update_row(
                                folder_name="usual_graphon",
                                file_name="method_comp" + f"_{folder_name}",
                                new_row=get_result_with_std(
                                    [sublist[crit_idx]+[criterion] for sublist in result], method=m),
                                row_index=idx,
                            )
                            idx += 1
                    else:
                        update_row(
                            folder_name="usual_graphon",
                            file_name="method_comp" + f"_{folder_name}",
                            new_row=get_result_with_std(result, method=m),
                            row_index=idx,
                        )
                        idx += 1
