import pickle

import numpy as np
from loguru import logger
import sys
import os
import traceback

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
from datetime import datetime
from sparsity_regime import scaling_functions

CRITERIONS = ["bic", "mdl"]#"aic", "mallows_cp", "hqic", "fpe", "elbow"]


def setup_logger_file(folder_name: str):
    logger.remove()
    logger.enable("pygraphon")
    logger.add(
        os.path.join(folder_name, "log_exp.txt"),
        level="TRACE",
        mode="w",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {file} | line {line} | {level} | {message}",
    )
    logger.add(
        os.path.join(folder_name, "log_errors.txt"),
        level="CRITICAL",
        mode="w",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {file} | line {line} | | {level} | {message}",
    )
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {file} | line {line} | {level} | {message}",
    )
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
    graphon_repr = graphon.repr if graphon is not None else ""
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
            logger.info("Finished fitting SmoothNetHist")
            est.get_ratio_par()
            for criterion in CRITERIONS:
                if graphon is not None:
                    norm_mse = MseProbaEdge(n_nodes=n)(
                        graphon, est.best_pairs[criterion]["graphon"]
                    )
                    norm_mae = MaeProbaEdge(n_nodes=n)(
                        graphon, est.best_pairs[criterion]["graphon"]
                    )
                else:
                    norm_mse = MseProbaEdge(n_nodes=n)(
                        adjacency_matrix, est.best_pairs[criterion]["graphon"]
                    )
                    norm_mae = MaeProbaEdge(n_nodes=n)(
                        adjacency_matrix, est.best_pairs[criterion]["graphon"]
                    )
                # Compute AUC and AUPRC
                norm_auc = AUCEdge(n_nodes=n)(adjacency_matrix, est.best_pairs[criterion]["pij"])
                norm_auprc = AUPRCEdge(n_nodes=n)(
                    adjacency_matrix, est.best_pairs[criterion]["pij"]
                )
                # if SmoothNetHist, compute BIC and ratio_par, leave None otherwise
                s = len(np.unique(est.best_pairs[criterion]["graphon"].graphon))
                graphon_repr = graphon.repr
                sub_result = [
                    method,
                    graphon_repr,
                    n,
                    norm_mse,
                    norm_mae,
                    norm_auc,
                    norm_auprc,
                    est.criterion_values[criterion],
                    est._num_par_smooth[criterion],
                ]
                result.append(sub_result)
        elif method == "HistogramEstimator":
            est.fit(adjacency_matrix)
            logger.info("Finished fitting HistogramEstimator")
            if graphon is not None:
                norm_mse = MseProbaEdge(n_nodes=n)(graphon, est.get_graphon())
                norm_mae = MaeProbaEdge(n_nodes=n)(graphon, est.get_graphon())
            else:
                norm_mse = MseProbaEdge(n_nodes=n)(adjacency_matrix, est.get_graphon())
                norm_mae = MaeProbaEdge(n_nodes=n)(adjacency_matrix, est.get_graphon())
            # Compute AUC and AUPRC
            norm_auc = AUCEdge(n_nodes=n)(adjacency_matrix, est.get_edge_connectivity())
            norm_auprc = AUPRCEdge(n_nodes=n)(adjacency_matrix, est.get_edge_connectivity())
            graphon_repr = graphon.repr
            result = [method, graphon_repr, n, norm_mse, norm_mae, norm_auc, norm_auprc, 0, 0]
        else:
            est.fit(adjacency_matrix)
            logger.info("Finished fitting " + method)
            if graphon is not None:
                norm_mse = MseProbaEdge(n_nodes=n)(graphon, est.get_graphon())
                norm_mae = MaeProbaEdge(n_nodes=n)(graphon, est.get_graphon())
            else:
                norm_mse = MseProbaEdge(n_nodes=n)(adjacency_matrix, est.get_graphon())
                norm_mae = MaeProbaEdge(n_nodes=n)(adjacency_matrix, est.get_graphon())
            # Compute AUC and AUPRC
            norm_auc = AUCEdge(n_nodes=n)(adjacency_matrix, est.get_edge_connectivity())
            norm_auprc = AUPRCEdge(n_nodes=n)(adjacency_matrix, est.get_edge_connectivity())
            result = [method, graphon_repr, n, norm_mse, norm_mae, norm_auc, norm_auprc, 0, 0]
    except Exception as e:
        logger.critical(f"Error with {method} on {graphon_repr} with n={n}")
        logger.critical(e)
        traceback.print_exc()
        if method == "SmoothNetHist":
            result = [[method, graphon_repr, n, np.nan, np.nan, np.nan, np.nan, 0, 0] for _ in CRITERIONS]
        else:
            result = [method, graphon_repr, n, np.nan, np.nan, np.nan, np.nan, 0, 0]
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
    n_min, n_max, n_step = 100, 2000, 100
    n_list = np.arange(n_min, n_max + 1, n_step)
    n_list = [300, 2000]

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Compute scaling factors if sparse
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    rho_common = 1  # Desired common value at n_min
    scaling_factors = {}
    for name, func in scaling_functions.items():
        g_n_min = func(n_min)
        c_k = rho_common / g_n_min
        scaling_factors[name] = c_k

    # Generate scaled rho_n lists
    scaled_rho_n = {}
    for name, func in scaling_functions.items():
        print(f'Scaling factor for {name}: {scaling_factors[name]}')
        c_k = scaling_factors[name]
        scaled_rho_n[name] = [c_k * func(n) for n in n_list]

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
                file_name="mult_sbm_ssm.csv",
                n_graphons=len(n_list) * len(k_list) * len(scaling_functions),
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
                graphon = StepGraphon(theta.Theta, bandwidthHist=1 / k)

                # Iterate over sparsity regimes
                for rho_name, rho_values in scaled_rho_n.items():
                    logger.info(f"Processing sparsity regime: {rho_name}")

                    for j, n in enumerate(n_list):
                        rho = rho_values[j]
                        logger.info(f"Applying rho_n={rho:.4f} for n={n}")
                        # Sparsify the graphon
                        graphon.sparsify(rho=rho)

                        result = []
                        for r in range(mult_rep):
                            logger.info(f"Repetition {r+1}/{mult_rep} for k={k}, n={n}, rho={rho_name}")
                            adj_matrix = graphon.draw(n=n, exchangeable=False)

                            # -- TRY/EXCEPT HERE --
                            try:
                                experiment_result = main(
                                    adjacency_matrix=adj_matrix,
                                    graphon=graphon,
                                    method="SmoothNetHist",
                                    folder_name=folder_name,
                                )
                                result.append(experiment_result)
                            except ValueError as e:
                                logger.warning(
                                    f"ValueError for k={k}, n={n}, repetition={r+1}, "
                                    f"rho={rho_name}. Storing NaNs. Error: {e}"
                                )
                                # 'SmoothNetHist' typically returns a list of length len(CRITERIONS),
                                # each item being some small list. We'll store a matching shape:
                                nan_list = [[np.nan] for _ in range(len(CRITERIONS))]
                                result.append(nan_list)
                                continue
                            # -- END TRY/EXCEPT --

                        # Update CSV with results for each criterion
                        for crit_idx, criterion in CRITERIONS:
                            update_row(
                                folder_name="sbm_ssm",
                                file_name="mult_sbm_ssm.csv",
                                new_row=get_result_with_std(
                                    [sublist[crit_idx] for sublist in result],
                                    method="SmoothNetHist",
                                    factor_name=rho_name
                                ),
                                row_index=(
                                    i * len(n_list) * len(scaling_functions) * len(CRITERIONS)
                                    + j * len(scaling_functions) * len(CRITERIONS)
                                    + list(scaling_functions.keys()).index(rho_name) * len(CRITERIONS)
                                    + crit_idx
                                    + 1
                                ),
                            )

        # ------------------------------------
        # ------- If single repetition -------
        # ------------------------------------
        else:
            csv_gen_graphon(
                path="results/sbm_ssm",
                file_name="comparison_sbm_ssm.csv",
                n_graphons=k_max * len(scaling_functions),
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

            for rho_name, rho_values in scaled_rho_n.items():
                logger.info(f"Processing sparsity regime: {rho_name}")

                for j, n in enumerate(n_list):
                    rho = rho_values[j]
                    logger.info(f"Applying rho_n={rho:.4f} for n={n}")
                    graphon.sparsify(rho=rho)

                    result = []
                    for r in range(mult_rep):
                        logger.info(f"Repetition {r+1}/{mult_rep} for k={k_max}, n={n}, rho={rho_name}")
                        adj_matrix = graphon.draw(n=n, exchangeable=False)

                        # -- TRY/EXCEPT HERE --
                        try:
                            experiment_result = main(
                                adjacency_matrix=adj_matrix,
                                graphon=graphon,
                                method="SmoothNetHist",
                                folder_name=folder_name,
                            )
                            result.append(experiment_result)
                        except ValueError as e:
                            logger.warning(
                                f"ValueError for k={k_max}, n={n}, repetition={r+1}, "
                                f"rho={rho_name}. Storing NaNs. Error: {e}"
                            )
                            nan_list = [[np.nan] for _ in range(len(CRITERIONS))]
                            result.append(nan_list)
                            continue
                        # -- END TRY/EXCEPT --

                    # Update CSV
                    update_row(
                        folder_name="sbm_ssm",
                        file_name="comparison_sbm_ssm.csv",
                        new_row=get_result_with_std(
                            result,
                            method="SmoothNetHist",
                            rho_name=rho_name
                        ),
                        row_index=1,  # Adjust as needed
                    )

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # If generate with other graphons
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    if folder_name == "usual_graphon":
        file_name = (
            "method_comp_subset_fn"
            + f"_{folder_name}"
            + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
            + ".csv"
        )
        graphons = []
        method = ["SmoothNetHist", "SAS", "USVT"]  # etc.
        list_graphons = ["assortative_sbm"]#, "dissassortative_sbm"]
        # e.g.: "hsbm", "hsbm_partition", "partition"

        for g in list_graphons:
            g = common_graphons[g]
            graphons.append(g)

        # Estimate total number of graphon-method-n-rho combinations
        total_graphons = len(graphons)
        total_methods = len(method)
        total_methods += len(CRITERIONS)-1 if "SmoothNetHist" in method else 0
        total_rhos = len(scaling_functions)
        total_n = len(n_list)
        total_entries = total_graphons * total_methods * total_rhos * total_n

        csv_gen_graphon(
            path="results/usual_graphon",
            file_name=file_name,
            n_graphons= total_entries,#len(graphons) * (len(method) + len(CRITERIONS) - 1) * len(n_list) + 1,
            smoothnethist_par=True,
        )
        idx = 0
        # Iterate over each graphon
        for g in graphons:
            # Iterate over each method
            for m in method:
                # Iterate over each sparsity regime
                for rho_name, rho_values in scaled_rho_n.items():
                    logger.info(f"Processing graphon={g.repr}, method={m}, sparsity regime={rho_name}")

                    # Iterate over each n
                    for k, n in enumerate(n_list):
                        rho = rho_values[k]
                        logger.info(f"Applying rho_n={rho:.4f} for n={n}")

                        # Copy the graphon to avoid in-place modification
                        g_copy = g
                        g_copy.sparsify(sparse_factor=rho)

                        result = []
                        for r in range(mult_rep):
                            logger.info(
                                f"Repetition {r+1}/{mult_rep} for graphon={g.repr}, "
                                f"method={m}, n={n}, rho={rho_name}"
                            )
                            np.random.seed(42 + k + (r + 1) * (len(n_list) + 1))
                            adj_mat = g_copy.draw(n=n, exchangeable=False)

                            # -- TRY/EXCEPT HERE --
                            try:
                                experiment_result = main(
                                    adjacency_matrix=adj_mat,
                                    graphon=g_copy,
                                    method=m,
                                    folder_name=folder_name,
                                )
                            except (ValueError, OverflowError) as e:
                                logger.warning(
                                    f"ValueError for graphon={g.repr}, n={n}, method={m}, "
                                    f"repetition={r+1}, rho={rho_name}. Storing NaNs. Error: {e}"
                                )
                                if m == "SmoothNetHist":
                                    # Return shape consistent with your usage of sublist[crit_idx] + [criterion]
                                    for _ in CRITERIONS:
                                        experiment_result = [[np.nan for _ in range(9)]]
                                else:
                                    # For other methods, often a single float or small list is returned
                                    # We'll store just [np.nan]
                                    experiment_result = [np.nan for _ in range(9)]
                                continue
                            result.append(experiment_result)
                            # -- END TRY/EXCEPT --

                        # Update CSV with results
                        if m == "SmoothNetHist":
                            # For each criterion, you do sublist[crit_idx] + [criterion]
                            for crit_idx, criterion in enumerate(CRITERIONS):
                                update_row(
                                    folder_name="usual_graphon",
                                    file_name=file_name,
                                    new_row=get_result_with_std(
                                        # Notice we select sublist[crit_idx]
                                        [sublist[crit_idx] + [criterion] for sublist in result],
                                        method=m,
                                        factor_name=rho_name
                                    ),
                                    row_index=idx,
                                )
                                idx += 1
                        else:
                            # For methods with a single metric per run
                            update_row(
                                folder_name="usual_graphon",
                                file_name=file_name,
                                new_row=get_result_with_std(
                                    result,
                                    method=m,
                                    factor_name=rho_name
                                ),
                                row_index=idx,
                            )
                            idx += 1
