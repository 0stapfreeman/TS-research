"""
Baseline imputation methods for comparison.

Simple methods that serve as baselines against the sparse recovery approaches.
"""

import numpy as np


def impute_mean(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fill missing values with column mean of observed values.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, m) where N = timesteps, m = number of series
    mask : np.ndarray
        Boolean mask of shape (N,), True = observed

    Returns
    -------
    np.ndarray
        Imputed data matrix of shape (N, m)
    """
    X_imputed = X.copy()
    miss = ~mask

    for j in range(X.shape[1]):
        col_mean = X[mask, j].mean()
        X_imputed[miss, j] = col_mean

    return X_imputed


def impute_linear(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fill missing values using linear interpolation per column.

    Interpolates between observed values. Extrapolates using nearest
    observed value for missing values at the boundaries.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, m) where N = timesteps, m = number of series
    mask : np.ndarray
        Boolean mask of shape (N,), True = observed

    Returns
    -------
    np.ndarray
        Imputed data matrix of shape (N, m)
    """
    N, m = X.shape
    X_imputed = X.copy()
    obs_idx = np.where(mask)[0]
    all_idx = np.arange(N)

    for j in range(m):
        # Use numpy interp which handles extrapolation with boundary values
        X_imputed[:, j] = np.interp(all_idx, obs_idx, X[obs_idx, j])

    return X_imputed


def impute_knn(X: np.ndarray, mask: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Fill missing values using K-nearest neighbors.

    Uses sklearn.impute.KNNImputer with k neighbors. The KNN imputation
    considers the correlation structure across all series.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, m) where N = timesteps, m = number of series
    mask : np.ndarray
        Boolean mask of shape (N,), True = observed
    k : int
        Number of neighbors to use (default 5)

    Returns
    -------
    np.ndarray
        Imputed data matrix of shape (N, m)
    """
    from sklearn.impute import KNNImputer

    # Create a copy with NaN for missing values
    X_with_nan = X.copy().astype(float)
    miss = ~mask
    X_with_nan[miss, :] = np.nan

    # Apply KNN imputation
    imputer = KNNImputer(n_neighbors=k)
    X_imputed = imputer.fit_transform(X_with_nan)

    return X_imputed
