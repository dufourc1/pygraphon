import io
import logging
import pickle

import numpy as np
from loguru import logger
from tqdm import tqdm
import os
import shutil

from .utils import parser_argument


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


def main(folder_name):

    logger.info("Starting experiment")
    # Initialise experiments
    # ----------------------
    metric_list = []

    # Run experiments
    # ----------------------

    logger.info("Computing done")
    with open("results"+f"{folder_name}.pkl", "wb") as f:
        pickle.dump(metric_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Cleaning {folder_name}/")
    # clean temp folder
    folder = 'experiments/results/'+f'{folder_name}'
    for filename in os.listdir(folder):
        if folder_name in filename:  # only delete files from corresponding regime
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
    # init logger because Charles corrupted me
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    logger.add("log_exp.txt", level="INFO", mode="w")

    # n init
    n_min, n_max, n_step = 50, 201, 50
    n_list = np.arange(n_min, n_max, n_step)

    # sparse regime import and init
    folder_name = "sbm_ssm"
    parser_argument(folder_name)

    # k init
    k_max = 5
    k_list = list(range(2, k_max + 1))
    # theta init
    theta_intra_min, theta_intra_max = 0.15, 0.5
    theta_inter_min, theta_inter_max = 0.05, 0.15

    main()
