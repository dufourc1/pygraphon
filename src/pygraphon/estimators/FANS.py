import numpy as np
from scipy.stats import kendalltau


def get_d0(A):
    A_sq = np.dot(A, A)
    d0 = np.triu(np.abs(A_sq[:, np.newaxis, :] - A_sq[np.newaxis, :, :]), k=1)
    d0 += np.random.uniform(0, 1, size=d0.shape) / A.shape[1]
    return np.nanmax(d0, axis=2) / A.shape[1]


def get_s(X):
    X_sq = np.dot(X, X.T)
    s = np.triu(np.abs(X_sq[:, np.newaxis, :] - X_sq[np.newaxis, :, :]), k=1)
    s += np.random.uniform(0, 1, size=s.shape) / X.shape[1]
    return np.nanmax(s, axis=2) / X.shape[1]


def get_cor(d0, s):
    d0_vec = d0[~np.isnan(d0)]
    s_vec = s[~np.isnan(s)]
    return kendalltau(s_vec, d0_vec)[0]


def FANS(A, X, _lambda, screen=False):
    n = A.shape[0]
    d0 = get_d0(A)
    s = get_s(X.astype(float))

    if screen:
        feature_cor = np.array([get_cor(d0, get_s(X[:, i].reshape(-1, 1))) for i in range(X.shape[1])])
        feature_ind = feature_cor > 0.05

        if np.sum(feature_ind) == 0:
            d = d0
            h = np.sqrt(np.log(n) / n)
            NB = np.where((d <= np.quantile(d[:, np.newaxis], h, axis=0)) & (np.eye(n) != 1), 1, np.nan)
            NB[np.sum(NB, axis=1) > 1, np.arange(n)] = 0
            kernel = NB
            W = np.dot(kernel, A) / np.dot(np.diag(np.sum(kernel, axis=1)), np.ones((n, n)))
            return (W + W.T) / 2

        s = get_s(X[:, feature_ind].astype(float))

    d = d0 / np.max(d0) + _lambda * s / np.max(s)
    h = np.sqrt(np.log(n) / n)
    NB = np.where((d <= np.quantile(d[:, np.newaxis], h, axis=0)) & (np.eye(n) != 1), 1, np.nan)
    NB[np.sum(NB, axis=1) > 1, np.arange(n)] = 0

    kernel = NB
    W = np.dot(kernel, A) / np.dot(np.diag(np.sum(kernel, axis=1)), np.ones((n, n)))
    return (W + W.T) / 2
