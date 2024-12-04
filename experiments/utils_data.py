import numpy as np
# from loguru
import csv
from typing import Optional
import shutil
import os


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


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
    path_to_file = os.path.join(THIS_FOLDER, path, file_name)
    column_names = [
        "Method",
        "Type of Graphon",
        "n",
        "MSE",
        "std(MSE)",
        "MAE",
        "std(MAE)",
        "AUC",
        "std(AUC)",
        "AUPRC",
        "std(AUPRC)",
        "BIC",
        "std(BIC)",
        "RatioPar",
        "std(RatioPar)"
    ]

    # Write the column names to the CSV file
    with open(path_to_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)

    # Append k empty lines to the CSV file
    with open(path_to_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for _ in range(n_graphons):
            writer.writerow([])

    # logger.debug(f"Empty CSV file '{path_to_file}' with {n_graphons} lines created.")


def csv_store_StepGraphon_parameters(
    path: str,
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
    path_to_file = os.path.join(THIS_FOLDER, path, file_name)
    if SBM_SSM:
        column_names = ["SBM", "n", "Theta", "Areas"]
    else:
        column_names = ["SSM", "n", "Theta", "Areas"]

    # Write the column names to the CSV file
    with open(path_to_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)

    # Append k empty lines to the CSV file
    with open(path_to_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for _ in range(n_it):
            writer.writerow([])

    # logger.debug(f"Empty CSV file '{path_to_file}' with {n_it} lines created.")


def update_row(file_name: str, folder_name: str, row_index: int, new_row: list) -> None:
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
    if not file_name.endswith(".csv"):
        file_name += ".csv"
    csv_file = os.path.join(THIS_FOLDER, "results",
                            folder_name, file_name)
    with open(csv_file, "r") as file:
        rows = list(csv.reader(file))
        rows[row_index] = new_row

    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def clean_temp_folder(folder_name: str) -> None:
    # clean temp folder
    folder = "results/" + f"{folder_name}"
    for filename in os.listdir(folder):
        if folder_name in filename:  # only delete files from corresponding regime
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))


def get_result_with_std(list_of_results: list, method=str, factor_name:str = None,) -> list:
    """Computes the mean and the standard deviation of a list of results.
    Creates a list of strings with the mean and the standard deviation.

    Parameters
    ----------
    list_of_results : list
        list of ndarray of results

    Returns
    -------
    list
        mean and standard deviation
    """
    result = []
    if method == 'SmoothNetHist':
        result.append(list_of_results[0][0]+factor_name+'_'+list_of_results[0][-1])
    else:
        result.append(list_of_results[0][0]+factor_name)
    result.append(list_of_results[0][1])
    result.append(list_of_results[0][2])
    selected_elements = [sublist[3:-1] for sublist in list_of_results]

    result_mean = np.mean(selected_elements, axis=0)
    result_stds = np.std(selected_elements, axis=0)
    for mean, std in zip(result_mean, result_stds):
        result.append(mean)
        result.append(std)
    return result
