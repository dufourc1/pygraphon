import numpy as np
import scipy

def two_d_bern_fl_pred(M_tour, lambda_grid, n):
    """
    Computes the best solution and the predicted likelihood for the Two-dimensional Bernoulli Fused Lasso problem.

    Parameters
    ----------
    M_tour : (n, n) ndarray
        Tour matrix.
    lambda_grid : 1d-array-like or list
        List of lambda values.
    n : int
        Size of the square matrix.

    Returns
    -------
    best_sol : (n, n) ndarray
        Array of the best solution.
    pred : list
        List of the predicted likelihood for each lambda value.

    Raises
    ------
    ValueError
        If the matrices `M_tour` or `lambda_grid` have incorrect shapes.

    """
    if not (M_tour.shape[0] == n and M_tour.shape[1] == n):
        raise ValueError("`M_tour` should be a 2D array of shape (n, n)")

    if not isinstance(lambda_grid, list):
        raise ValueError("`lambda_grid` should be a 1D array-like of shape (length)")

    #df = [0] * len(lambda_grid)
    pred = [0] * len(lambda_grid)
    #minus_lik = [0] * len(lambda_grid)

    non_missing = np.where(np.random.rand(n, n) < 0.8, 1, 0)
    M_obs = np.where(non_missing > 0, M_tour, 0)
    missing = 1 - non_missing

    best_sol = np.zeros((n, n))
    best_pred = np.Inf

    indices = np.where(missing > 0)

    for j in range(len(lambda_grid)):

        if j == 0:
            temp = fusedlasso_weightedl2_admm2d(M_obs, lambda_=lambda_grid[j])

        if j > 0:
            temp = fusedlasso_weightedl2_admm2d(M_obs, lambda_=lambda_grid[j], x_init=temp['x'])

        theta_hat = np.array(temp['x']).reshape(-1,)

        theta_hat_mat = np.array(theta_hat).reshape((n, n))

        pred[j] = -np.sum(np.log(scipy.stats.binom.pmf(M_tour[indices], 1, theta_hat_mat[indices])))

        if pred[j] < best_pred:

            best_pred = pred[j]
            best_sol = theta_hat_mat

    best_ind = np.argmin(pred)

    temp = fusedlasso_weightedl2_admm2d(M_tour, lambda_=lambda_grid[best_ind], x_init=temp['x'])

    best_sol = temp['x']

    return {"best_sol": best_sol, "pred": pred}


def fusedlasso_weightedl2_admm2d(y, w=None, lamb=1, iter_max=500, rel_tol=1e-3, inflate=1.5, x_init=None):
    """
    Solves the fused lasso on a 2D grid graph under a weighted l2 loss.

    Parameters
    ----------
    y : (n, m) ndarray
        Matrix of observations.
    w : (n, m) ndarray, optional
        Weight matrix. If not provided, the matrix of all ones is used.
    lamb : float, optional
        Penalty parameter.
    iter_max : int, optional
        Maximum number of iterations to perform.
    rel_tol : float, optional
        Convergence tolerance.
    inflate : float, optional
        Factor to modify the penalty parameter for the dual, based on the primal and dual residuals.
    x_init : (n, m) ndarray, optional
        Initial value for the variable to be optimized. If not provided, the mean of `y` is used.

    Returns
    -------
    result : dict
        Dictionary with the following keys:

        - `x`: (n, m) ndarray, the optimized variable.
        - `z`: (n, m) ndarray, slack variable for the likelihood.
        - `u`: (n, m) ndarray, scaled dual variable for the constraint.
        - `primal_trace`: list
            List of the primal residuals' norms for each iteration.
        - `dual_trace`: list
            List of the dual residuals' norms for each iteration.
        - `counter`: int
            Counter of the number of iterations.

    Notes
    -----
    This function implements the Fused Lasso on a 2D grid graph under a weighted l2 loss. The algorithm used is a variation of the Alternating Direction Method of Multipliers (ADMM).

    References
    ----------
    .. [1] Brendt Wirth, Martin Genkin, and Sarel J. Steel. "The fused lasso for grid-structured data with an application to MALDI-TOF mass spectrometry."
       Statistics and Computing 23.2 (2013): 255-270.

    Examples
    --------
    Create a synthetic problem with a 2D grid graph.

    >>> import numpy as np
    >>> n, m = 10, 10
    >>> y = np.random.rand(n, m)
    >>> w = np.ones_like(y)
    >>> lamb = 1
    >>> result = fusedlasso_weightedl2_admm2d(y, w, lamb)
    >>> print(result['x'].shape)
    (10, 10)
    """
    # Initialization
    if x_init is None:
        x_init = np.mean(y) * np.ones((len(y), len(y[0])))
    x = x_init

    z = np.zeros((len(y), len(y[0])))
    u = np.zeros((len(y), len(y[0])))
    a = 2 * lamb
    if w is None:
        w = np.ones((len(y), len(y[0])))

    primal_trace = []
    dual_trace = []
    converged = False
    counter = 0
    while not converged and counter < iter_max:
        this_y = (w * y + a * (z - u)) / (w + a)

        for i in range(len(y)):
            x[i] = fl_dp_weight(this_y[i], a + w[i], lamb)

        z_new = z
        for i in range(len(y[0])):
            z_new[:, i] = fl_dp(x[:, i] + u[:, i], lamb)

        dual_residual = a * (z_new - z)
        z = z_new

        primal_residual = x - z
        u = u + primal_residual

        primal_resnorm = np.linalg.norm(primal_residual, ord=2)
        dual_resnorm = np.linalg.norm(dual_residual, ord=2)
        if dual_resnorm < rel_tol and primal_resnorm < rel_tol:
            converged = True

        if primal_resnorm > 5 * dual_resnorm:
            a = inflate * a
            u = u / inflate
        elif dual_resnorm > 5 * primal_resnorm:
            a = a / inflate
            u = inflate * u

        primal_trace.append(primal_resnorm)
        dual_trace.append(dual_resnorm)
        counter += 1

    return {'x': x, 'z': z, 'u': u, 'primal_trace': primal_trace, 'dual_trace': dual_trace, 'counter': counter}


def fl_dp_weight(y, a, lamb):
    """
    Implements the proximal map of the weighted lasso penalty function.

    Parameters
    ----------
    y : (n,) ndarray
        Vector of input values.
    a : float
        Scalar parameter.
    lamb : float
        Penalty parameter.

    Returns
    -------
    out : (n,) ndarray
        Vector of proximal map evaluations.

    Notes
    -----
    Implementation of the function `f(x) = a*x + sign(x)*abs(x)` in the proximal map
    for the weighted lasso penalty.

    Examples
    --------
    Create a simple example of the input values.

    >>> import numpy as np
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> a = 1
    >>> lamb = 2
    >>> out = fl_dp_weight(y, a, lamb)
    >>> print(out)
    [0. 1. 1. 2. 2.]
    """

    n = len(y)
    out = np.zeros(n)
    for i in range(n):
        if y[i] > lamb / a:
            out[i] = y[i] - lamb / a
        elif y[i] < -lamb / a:
            out[i] = y[i] + lamb / a
        else:
            out[i] = 0
    return out

def fl_dp(y, lamb):
    """
    Implements the soft-thresholding function in the proximal map for the fused lasso penalty.

    Parameters
    ----------
    y : (n,) ndarray
        Vector of input values.
    lamb : float
        Penalty parameter.

    Returns
    -------
    out : (n,) ndarray
        Vector of proximal map evaluations.

    Notes
    -----
    Implementation of the soft-thresholding function in the proximal map for the fused lasso penalty.

    Examples
    --------
    Create a simple example of the input values.

    >>> import numpy as np
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> lamb = 2
    >>> out = fl_dp(y, lamb)
    >>> print(out)
    [1. 2. 1. 2. 1.]
    """

    out = np.maximum(y - lamb, 0) - np.maximum(-y - lamb, 0)
    return out

