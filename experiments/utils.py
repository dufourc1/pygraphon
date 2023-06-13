import argparse
import numpy as np
from loguru import logger
import csv
from typing import Optional
import os


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


def parser_argument(folder_name: str) -> None:
    """Gives the possibility to run the experiment from the command line, changing

    Parameters
    ----------
    folder_name : str
        folder to save the results
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", default=folder_name)
    args = parser.parse_args()
    logger.info(args.regime)
    logger.info(type(args.regime))


def csv_gen_graphon(
        path: str,
        file_name: str,
        n_graphons: int,
        smoothnethist_par: Optional[bool] = False,
        ) -> None:
    """Generates the csv file to store results when testing estimator for different graphons.

    Parameters
    ----------
    file_name : str
        file to save the results
    n_graphons : int
        number of graphons tested
    smoothnethist_par : bool, optional
        if True, use the smoothnethist parameter, by default False
    """
    if not file_name.endswith(".csv"):
        file_name = file_name + ".csv"
    path_to_file = os.path.join(path, file_name) 
    if smoothnethist_par:
        column_names = ["Type of Graphon", "n", "MSE", "MAE", "AUC", "AUPRC", "BIC", "RatioPar"]
    else:
        column_names = ["Type of Graphon", "n", "MSE", "MAE", "AUC", "AUPRC"]

    # Write the column names to the CSV file
    with open(path_to_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)

    # Append k empty lines to the CSV file
    with open(path_to_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for _ in range(n_graphons):
            writer.writerow([])

    print(f"Empty CSV file '{path_to_file}' with {n_graphons} lines created.")

def csv_store_StepGraphon_parameters(path: str, 
                                     file_name: str, 
                                     n_it: int,
                                     SBM_SSM: bool,
                                     ) -> None:
    """Stores the parameters of the Step Graphon in a csv file.

    Parameters
    ----------
    path : str
        path to the csv file
    file_name : str
        name of the csv file
    n_it : int
        number of iterations
    SSM_SBM : bool
        if True, the graphon is a SBM, else it is a SSM
    """
    if not file_name.endswith(".csv"):
        file_name = file_name + ".csv"
    path_to_file = os.path.join(path, file_name)
    if SBM_SSM:
        column_names = ["SBM", "n", "Theta", "Areas"]
    else:
        column_names = ["SSM", "n", "Theta", "Areas"]

    # Write the column names to the CSV file
    with open(path_to_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)

    # Append k empty lines to the CSV file
    with open(path_to_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for _ in range(n_it):
            writer.writerow([])

    print(f"Empty CSV file '{path_to_file}' with {n_it} lines created.")



def update_row(file_name : str,
               folder_name : str,
               row_index : int,
               new_row: list
               ) -> None:
    """Updates a row in a csv file.

    Parameters
    ----------
    file_name : str
        name of the csv file
    folder_name : str
        name of the folder containing the csv file
    row_index : int
        index of the row to update
    new_row : list
        new row
    """

    csv_file = os.path.join(THIS_FOLDER, 'results', folder_name, file_name + '.csv')
    with open(csv_file, 'r') as file:
        rows = list(csv.reader(file))
        rows[row_index] = new_row

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
