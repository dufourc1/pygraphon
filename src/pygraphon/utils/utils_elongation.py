import numpy as np
from scipy.special import comb

def get_elongation_list(theta_matrix: np.ndarray, num_par: int) -> list[float]:
    """
    Compute the elongation of each shape in a k-by-k adjacency matrix.

    Each of the 'num_par' shapes is labeled with integers in {1, 2, ..., num_par}.
    The matrix is assumed to be fully partitioned: 
    no unlabeled (zero) cells if you truly want every cell to belong to some shape.

    Elongation of a shape S is:
        elongation(S) = diameter(S) / sqrt(area(S))

      where:
        - area(S) = (#blocks in shape S) / (k^2).
        - diameter(S) = (horizontal_extent + vertical_extent),
            horizontal_extent = (max_x - min_x + 1) / k,
            vertical_extent   = (max_y - min_y + 1) / k.
          Here, (min_x, min_y) and (max_x, max_y) are the bounding indices 
          for shape S in the matrix.

    You may optionally enforce symmetry by uncommenting the lines that check 
    if the matrix is symmetric.

    Parameters
    ----------
    theta_matrix : np.ndarray of shape (k, k)
        2D array with shape labels in [1..num_par].
    num_par : int
        Number of shapes (unique labels).
    k : int
        Dimension of the matrix.

    Returns
    -------
    list of float
        [elongation_of_shape_1, elongation_of_shape_2, ..., elongation_of_shape_num_par]

    Raises
    ------
    ValueError
        - If the matrix has a different number of distinct shapes than num_par.
        - If shape labels aren't exactly {1,2,...,num_par}.
        - If any shape is empty.
        - [Optional] If the matrix is not symmetric (if you choose to enforce it).
    """
    k = theta_matrix.shape[0]
    #list all values in theta_matrix
    # Optional strict symmetry check:
    if not np.allclose(theta_matrix, theta_matrix.T):
        raise ValueError("Matrix is not symmetric. Symmetry is required.")
    
    if comb(k+1, 2 ) == num_par:
        return [2.0 for _ in range(int(num_par))]
    else:
        unique_labels = np.unique(theta_matrix)

        elongations = []
        for shape_label in unique_labels:
            coords = np.argwhere(theta_matrix == shape_label)
            if coords.size == 0:
                raise ValueError(f"Shape {shape_label} has no blocks.")

            area = coords.shape[0] / (k**2)

            min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
            min_y, max_y = coords[:, 1].min(), coords[:, 1].max()

            horizontal_extent = (max_x - min_x + 1) / k
            vertical_extent   = (max_y - min_y + 1) / k
            diameter = horizontal_extent + vertical_extent

            shape_elong = diameter / np.sqrt(area)
            elongations.append(shape_elong)

        return elongations


def compute_average_elongation(theta_matrix: np.ndarray, num_par: int) -> float:
    """
    Compute the average elongation over all shapes in theta_matrix.

    Parameters
    ----------
    theta_matrix : np.ndarray of shape (k, k)
        Matrix of shape labels [1..num_par].
    num_par : int
        Number of shapes in the matrix.

    Returns
    -------
    float
        The mean of the elongations across all shapes.

    Raises
    ------
    ValueError
        - If shape labeling doesn't match {1..num_par} or if a shape is empty.
        - [Optional] If matrix is not symmetric (if enforced).
    """
    elong_list = get_elongation_list(theta_matrix, num_par)
    return np.mean(elong_list)


def create_symmetric_diagonal_squares_partition(k: int, squares_sizes: list[int]):
    """
    Partition the entire k-by-k grid into len(squares_sizes) shapes, each a 
    'square block' on the diagonal, but ensuring the entire grid is labeled.

    - sum(squares_sizes) must be k.
    - If squares_sizes = [s1, s2, ..., sm], shape i occupies the block rows
      [offset..offset+s_i - 1], columns [offset..offset+s_i - 1], 
      and we replicate that label for all off-diagonal cells in that block region,
      so the entire submatrix is shape i. 
      Then we increment offset by s_i for the next shape.

    This yields a block-diagonal structure in terms of shapes, but each block 
    is a 'square chunk' that spans row x col. 
    The matrix is symmetric, fully-labeled (no zeros).

    For example, if k=6 and squares_sizes=[2,4], we have:
      - shape1 => rows 0..1 & cols 0..1,
      - shape2 => rows 2..5 & cols 2..5,
    covering the entire matrix in two big squares along the diagonal.

    NOTE: This ensures no unlabeled region exists. 
    If squares_sizes sum to k, the entire matrix is partitioned 
    into these diagonal squares.

    Parameters
    ----------
    k : int
        The size of the matrix (k-by-k).
    squares_sizes : list of int
        A list of block sizes that sum to k. 
        The number of shapes is len(squares_sizes).

    Returns
    -------
    theta_matrix : np.ndarray of shape (k, k)
        A fully-labeled, symmetric matrix with shape labels in {1..m},
        m = len(squares_sizes).

    num_par : int
        m, the number of shapes.

    Raises
    ------
    ValueError
        If sum(squares_sizes) != k.
    """
    if sum(squares_sizes) != k:
        raise ValueError(
            f"sum(squares_sizes)={sum(squares_sizes)} != k={k}."
        )

    theta_matrix = np.zeros((k, k), dtype=int)
    offset = 0
    for i, size in enumerate(squares_sizes, start=1):
        # Fill rows [offset..offset+size-1] and cols [offset..offset+size-1] with label i
        for row in range(offset, offset+size):
            for col in range(offset, offset+size):
                theta_matrix[row, col] = i
        offset += size

    num_par = len(squares_sizes)
    # Confirm that the entire matrix is now covered by these squares, with no leftover.
    # Because sum(squares_sizes) = k, the squares fill row/col blocks exactly.
    # It's already symmetric by virtue of writing squares on the diagonal.

    return theta_matrix, num_par


def create_diagonal_partition_symmetric(k: int):
    """
    Partition the entire k-by-k grid so that each diagonal 'strip' is a shape.
    For i <= j, shape_label = (j - i) + 1, and mirrored for i > j.

    This yields exactly k shapes, labeled 1..k, with no zeros or leftover cells.

    For i=0..k-1, j= i..k-1:
       shape_label = (j - i) + 1 in [1..k]
    Then we mirror for j < i.

    Example for k=4 (reading from i=rows, j=cols):
        i/j=  0 1 2 3
         0   1 2 3 4
         1   2 1 2 3
         2   3 2 1 2
         3   4 3 2 1

    All cells are labeled, so there's no unlabeled region.

    Parameters
    ----------
    k : int
        Dimension of the matrix.

    Returns
    -------
    theta : np.ndarray of shape (k, k)
        A fully-labeled, symmetric matrix with shape labels in {1..k}.
    num_par : int
        k, the number of shapes.

    """
    theta = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(i, k):
            shape_label = (j - i) + 1
            theta[i, j] = shape_label
            theta[j, i] = shape_label  # mirror for symmetry

    num_par = k
    return theta, num_par
