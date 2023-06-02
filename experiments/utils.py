import argparse
from loguru import logger
import csv
from typing import Optional


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
        folder_name: str,
        n_graphons: int,
        smoothnethist_par: Optional[bool] = False,
        ) -> None:
    """Generates the csv file to store results when testing estimator for different graphons.

    Parameters
    ----------
    folder_name : str
        folder to save the results
    n_graphons : int
        number of graphons tested
    smoothnethist_par : bool, optional
        if True, use the smoothnethist parameter, by default False
    """

    filename = f'{folder_name}.csv'
    if smoothnethist_par:
        column_names = ["Type of Graphon", "n", "MSE", "MAE", "AUC", "AUPRC", "BIC", "RatioPar"]
    else:
        column_names = ["Type of Graphon", "n", "MSE", "MAE", "AUC", "AUPRC"]

    # Write the column names to the CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)

    # Append k empty lines to the CSV file
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for _ in range(n_graphons):
            writer.writerow([])

    print(f"Empty CSV file '{filename}' with {n_graphons} lines created.")
