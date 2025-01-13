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


def get_result_with_std(list_of_results: list, method=str, factor_name: str = None) -> list:
    """Computes the mean and the standard deviation of a list of results.
    Creates a list of strings with the mean and the standard deviation.
    
    Parameters
    ----------
    list_of_results : list
        list of ndarray (or list) of results
    method : str
        The method used (e.g. 'SmoothNetHist')
    factor_name : str
        Additional string appended to the first results in the output

    Returns
    -------
    list
        The combined list, including name information, single-run metrics,
        plus mean and std for the rest.
    """
    # Check factor_name is a string, otherwise raise error
    if not isinstance(factor_name, str):
        raise ValueError("factor_name must be a string")

    # Check that the first element is a string (as you specified)
    if not isinstance(list_of_results[0][0], str):
        raise ValueError("First element of list_of_results must be a string")

    # Prepare the result list
    result = []

    # Depending on the method, we build a name-string in result[0]
    if method == 'SmoothNetHist':
        # We assume: list_of_results[0][-1] is also a string
        result.append(list_of_results[0][0] + factor_name + '_' + list_of_results[0][-1])
    else:
        # For other methods
        result.append(list_of_results[0][0] + factor_name)
    
    # Next, we append some fixed items that appear in positions 1 and 2
    result.append(list_of_results[0][1])
    result.append(list_of_results[0][2])

    # 'selected_elements' are the numeric columns from index 3 up to (but excluding) the last
    # for each sublist in list_of_results
    selected_elements = [sublist[3:-1] for sublist in list_of_results]

    # Convert to numpy array for easier NaN checks
    # If sublists can differ in length, this might fail. We assume uniform length:
    selected_array = np.array(selected_elements, dtype=float)

    # If there's any NaN or inf in the data, we'll produce all-NaN for mean & std
    # (or you can choose a more selective approach if desired).
    if np.isnan(selected_array).any() or np.isinf(selected_array).any():
        # shape[1] is how many numeric columns we have
        n_cols = selected_array.shape[1] if selected_array.ndim > 1 else 0
        result_mean = [np.nan] * n_cols
        result_stds = [np.nan] * n_cols
    else:
        # Compute mean/std across rows
        result_mean = np.mean(selected_array, axis=0)
        result_stds = np.std(selected_array, axis=0)

    # Append each mean and std to the result list
    # If you have N columns, you'll add 2*N items here.
    for mean_val, std_val in zip(result_mean, result_stds):
        result.append(mean_val)
        result.append(std_val)

    return result

