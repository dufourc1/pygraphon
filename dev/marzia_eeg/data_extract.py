import scipy.io
import pandas as pd
import os
import numpy as np


def extract_wdpli(base_folder_path: str, surv=True):# -> tuple[dict, list[str]]:
    """Extracts the wdpli data from the folder path and returns a pandas
    dataframe.

    Parameters
    ----------
    folder_path : str
        path to the folder containing the wdpli data
    surv : bool, optional
        extract data from surviving (True) or non surviving (False),
        by default True

    Returns
    -------
    dict
        dictionary containing the extracted wdpli data
    list
        list of the folders from which the data was extracted
    """
    print(f"Extracting wdpli data from {base_folder_path}")
    dataframes_wpli = {}
    mat_files = ["dwPLI_1_40Hz"]
    # extract folders name from the path as a list os strings
    folders = []

    for folder in os.listdir(base_folder_path):
        if os.path.isdir(os.path.join(base_folder_path, folder)):
            folders.append(folder)
    for folder in folders:
        folder_path = os.path.join(base_folder_path, folder)
        for mat_file in mat_files:
            file_path = os.path.join(folder_path, mat_file + ".mat")
            if os.path.exists(file_path):
                mat_data = scipy.io.loadmat(file_path)
                data_key = 'dwpli'
                if data_key in mat_data:
                    data = mat_data[data_key]
                    if data.shape == (1, 1) and data.dtype.fields is not None:
                        if 'wpli_debiasedspctrm' in data.dtype.names:
                            wpli_data = data['wpli_debiasedspctrm'][0, 0]
                            for i in range(wpli_data.shape[2]):
                                slice_df = pd.DataFrame(wpli_data[:62, :62, i])
                                dataframes_wpli[f"{folder}_{mat_file}_wpli_debiasedspctrm_slice{i+1}"] = slice_df
                        else:
                            print(f"'wpli_debiasedspctrm' field not found in '{file_path}'")
                    else:
                        print(f"Unexpected data structure in '{file_path}': {data.shape}, {data.dtype}")
                else:
                    print(f"Data key '{data_key}' not found in '{file_path}'")
            else:
                print(f"File not found: {file_path}")
    return dataframes_wpli, folders


def create_adjacency_matrix(similarity_matrix, threshold):
    """Creates an adjacency matrix from a similarity matrix by thresholding.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        2-dimensional similarity matrix
    threshold : float
        threshold value for adjacency matrix

    Returns
    -------
    np.ndarray
        adjacency matrix
    """
    if similarity_matrix.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional")
    adjacency_matrix = np.where(similarity_matrix > threshold, 1, 0)
    np.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix


def adj_percentile_thresholding(similarity_matrix, percentile):
    """Creates an adjacency matrix from a similarity matrix by thresholding according to a percentile.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        2-dimensional similarity matrix
    percentile : float
        percentile value for thresholding, e.g. 90 for 90th percentile

    Returns
    -------
    np.ndarray
        adjacency matrix
    """
    threshold = np.nanpercentile(similarity_matrix, percentile)
    return create_adjacency_matrix(similarity_matrix, threshold)


def adj_brute_thresholding(similarity_matrix, threshold):
    """Creates an adjacency matrix from a similarity matrix by value-thresholding.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        2-dimensional similarity matrix
    threshold : float
        threshold value for adjacency matrix, e.g. 0.5

    Returns
    -------
    np.ndarray
        adjacency matrix
    """
    return create_adjacency_matrix(similarity_matrix, threshold)
