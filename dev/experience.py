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
import os, shutil
import time

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

    temp_folder = time.strftime("%Y%m%d-%H%M%S")

    logger.info("Starting experiment")
    # Initialise parameters: n nodes, k blocks, theta prob
    n_min, n_max, n_step =  50, 1001, 50
    n_list = np.arange(n_min, n_max, n_step)
    rho_n =0.2*np.power(n_min,1/4)/np.power(n_list, 1/4)
    #10/n_list
    #0.2*(n_min/np.log(n_min))*np.log(n_list)/n_list
    #0.2*(n_min/np.power(np.log(n_min),3))*np.power(np.log(n_list),3)/n_list
    #0.2*np.sqrt(n_min)/np.sqrt(n_list)
    #0.2*np.power(n_min,1/4)/np.power(n_list, 1/4)
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
            temp_folder,
        )
        for index in tqdm(it_list, file=tqdm_out)
    )

    #post process result (list of metric lists)
    for i,index in enumerate(it_list):
        print(i,index)
        for result_index in range(4):
            metric_list[result_index][index[0][0], index[1][0]] = result[i][result_index]
    

    logger.info("Computing done")
    #results name: 'results', 'results1n', 'resultslognn', 'resultslog3n', 'resultssqrt', 'resultssqrt4'
    with open("resultssqrt4.pkl", "wb") as f:
        pickle.dump(metric_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Cleaning {temp_folder}/")    
    #clean temp folder
    folder = './' + temp_folder
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


        logger.info("Results saved, bye")


if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    logger.add("log_exp.txt", level="INFO", mode="w")
    main()
