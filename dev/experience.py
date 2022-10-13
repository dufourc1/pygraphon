import itertools

import matplotlib.pyplot as plt
import numpy as np
from experiment_smoothing import smooth_exp_parallel
from gen_sbm import AssociativeFullyRandom
from joblib import Parallel, delayed
import pickle
import time
from loguru import logger
def main():

    logger.info("Starting experiment")
    #Initialise parameters: n nodes, k blocks, theta prob
    n_min, n_max, n_step = 60, 1500, 50
    n_list = np.arange(n_min, n_max, n_step)
    rho_n = 1/3

    k_max = 15
    k_list = list(range(2, k_max+1))
    theta_intra_min, theta_intra_max = 0.15, 0.5
    theta_inter_min, theta_inter_max = 0.05, 0.15

    #Create arrays to store metrics
    hm_matrix_smoothing_test_error = np.zeros((n_list.shape[0], len(k_list)))
    hm_matrix_smoothing_test_deviation_from_true = np.zeros_like(
        hm_matrix_smoothing_test_error)
    hm_matrix_smoothing_test_deviation_from_hist = np.zeros_like(
        hm_matrix_smoothing_test_error)
    hm_matrix_smoothing_test_comp_to_hist_perf = np.zeros_like(
        hm_matrix_smoothing_test_error)
    metric_list = [hm_matrix_smoothing_test_comp_to_hist_perf,
                hm_matrix_smoothing_test_error,
                hm_matrix_smoothing_test_deviation_from_hist,
                hm_matrix_smoothing_test_deviation_from_true]

    #Initialise connectivity matrices for different prior to nested loop
    #to store them and analyse them separately
    theta =[]
    for j, k in enumerate(k_list):
        theta.append(AssociativeFullyRandom(K=k,
                                    min_inside_classes=theta_intra_min,
                                    max_inside_classes=theta_intra_max,
                                    min_outside_classes=theta_inter_min,
                                    max_outside_classes=theta_inter_max).Theta)
    it_list = list(itertools.product(list(enumerate(n_list)), list(enumerate((k_list)))))
    result= Parallel(n_jobs=36, prefer="threads")(delayed(smooth_exp_parallel)(index[0][0], index[1][0], index[1][1], index[0][1], rho_n, theta[index[1][0]], metric_list) for index in it_list)

    logger.info("Computing done")
    with open("result.pkl", "wb") as f:
        pickle.dump(result,f, protocol = pickle.HIGHEST_PROTOCOL)
    logger.info("Results saved, bye")


if __name__ == "__main__":
    logger.add("log_exp.txt", level = "INFO", mode = "w")
    main()
