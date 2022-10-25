import io
import itertools
import logging
import pickle

import numpy as np
from experiment_smoothing import smooth_exp_parallel
from gen_sbm import AssociativeFullyRandom
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm


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


def main():

    logger.info("Starting experiment")
    # Initialise parameters: n nodes, k blocks, theta prob
    n_min, n_max, n_step =  50, 1001, 50
    n_list = np.arange(n_min, n_max, n_step)
    rho_n = 10/n_list

    k_max = 10
    k_list = list(range(2, k_max + 1))
    theta_intra_min, theta_intra_max = 0.15, 0.5
    theta_inter_min, theta_inter_max = 0.05, 0.15

    # Create arrays to store metrics
    error = np.zeros((n_list.shape[0], len(k_list)))
    deviation_from_true = np.zeros_like(error)
    deviation_from_hist = np.zeros_like(error)
    comp_to_hist_perf = np.zeros_like(error)
    metric_list = [
        comp_to_hist_perf,
        error,
        deviation_from_hist,
        deviation_from_true,
    ]

    # Initialise connectivity matrices for different prior to nested loop
    # to store them and analyse them separately
    theta = []
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    for j, k in enumerate(k_list):
        theta.append(
            AssociativeFullyRandom(
                K=k,
                min_inside_classes=theta_intra_min,
                max_inside_classes=theta_intra_max,
                min_outside_classes=theta_inter_min,
                max_outside_classes=theta_inter_max,
            ).Theta
        )
    it_list = list(itertools.product(list(enumerate(n_list)), list(enumerate((k_list)))))
    result = Parallel(n_jobs=36, prefer="threads")(
        delayed(smooth_exp_parallel)(
            index[0][0],#index of n in n_list
            index[1][0],#index of k in k_list
            index[1][1],#value of k
            index[0][1],#value of n
            rho_n[index[0][0]],#rho_n corresponding to correct n
            theta[index[1][0]],
        )
        for index in tqdm(it_list, file=tqdm_out)
    )

    #post process result (list of metric lists)
    for i,index in enumerate(it_list):
        for result_index in range(4):
            metric_list[result_index][index[0][0], index[1][0]] = result[i][result_index]
    

    logger.info("Computing done")
    with open("result.pkl", "wb") as f:
        pickle.dump(metric_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Results saved, bye")


if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    logger.add("log_exp.txt", level="INFO", mode="w")
    main()
