"""
Joint Imputation for Multivariate Time Series

Core functions for comparing joint (ℓ2,1) vs independent (ℓ1) sparse recovery
using DCT basis for time series imputation.
"""

import numpy as np
import cvxpy as cp
from scipy import fftpack
from joblib import Parallel, delayed


def build_dct_basis(N: int) -> np.ndarray:
    """Build orthonormal DCT-II basis matrix."""
    return fftpack.dct(np.eye(N), type=2, norm="ortho", axis=0)


def recover_joint(A: np.ndarray, Y: np.ndarray, delta: float, solver: str = "ECOS") -> np.ndarray:
    """
    Joint group-sparse recovery (shared-support) using ℓ2,1 norm.

    Solves: min_C  sum_k ||C[k,:]||_2  s.t. ||A @ C - Y||_F <= delta

    This encourages all series to use the same frequency components.
    """
    N = A.shape[1]
    m = Y.shape[1]
    C = cp.Variable((N, m))

    obj = cp.Minimize(cp.sum(cp.norm(C, 2, axis=1)))
    cons = [cp.norm(A @ C - Y, "fro") <= delta]
    prob = cp.Problem(obj, cons)

    if solver.upper() == "ECOS":
        prob.solve(solver=cp.ECOS, verbose=False)
    else:
        prob.solve(solver=cp.SCS, max_iters=5000, eps=1e-4, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"CVXPY failed (joint): {prob.status}")

    return np.asarray(C.value)


def recover_independent(A: np.ndarray, y: np.ndarray, delta: float, solver: str = "ECOS") -> np.ndarray:
    """
    Independent sparse recovery using ℓ1 norm for a single series.

    Solves: min_c ||c||_1  s.t. ||A @ c - y||_2 <= delta
    """
    N = A.shape[1]
    c = cp.Variable(N)

    obj = cp.Minimize(cp.norm1(c))
    cons = [cp.norm2(A @ c - y) <= delta]
    prob = cp.Problem(obj, cons)

    try:
        if solver.upper() == "ECOS":
            prob.solve(solver=cp.ECOS, verbose=False)
        else:
            prob.solve(solver=cp.SCS, max_iters=5000, eps=1e-4, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, max_iters=8000, eps=1e-4, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"CVXPY failed (independent): {prob.status}")

    return np.asarray(c.value)


def run_imputation_experiment(
    X: np.ndarray,
    keep_prob: float = 0.7,
    delta_rel: float = 1e-2,
    seed: int = 0,
    solver: str = "ECOS",
) -> dict:
    """
    Run joint vs independent imputation experiment on multivariate time series.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, m) where N = timesteps, m = number of series
    keep_prob : float
        Fraction of timesteps to keep as observed (default 0.7)
    delta_rel : float
        Relative noise tolerance (default 0.01)
    seed : int
        Random seed for reproducibility
    solver : str
        CVXPY solver to use ("ECOS" or "SCS")

    Returns
    -------
    dict with keys:
        - X_joint: Joint recovery result
        - X_ind: Independent recovery result
        - mask: Boolean mask of observed timesteps
        - rmse_joint: RMSE on missing values (joint)
        - rmse_ind: RMSE on missing values (independent)
    """
    N, m = X.shape

    # Standardize per series
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    X_norm = (X - mu) / sd

    # Create missing mask (same timesteps missing across all series)
    rng = np.random.default_rng(seed)
    mask = rng.random(N) < keep_prob
    obs_idx = np.where(mask)[0]

    # Build measurement matrix
    B = build_dct_basis(N)
    A = B[obs_idx, :]
    Y = X_norm[obs_idx, :]

    # Joint recovery
    delta_joint = delta_rel * np.linalg.norm(Y, "fro")
    C_joint = recover_joint(A, Y, delta=delta_joint, solver=solver)
    X_joint_norm = B @ C_joint

    # Independent recovery (per series) - parallelized
    def solve_one(j):
        delta_j = delta_rel * np.linalg.norm(Y[:, j])
        return recover_independent(A, Y[:, j], delta=delta_j, solver=solver)

    results = Parallel(n_jobs=-1)(delayed(solve_one)(j) for j in range(m))
    C_ind = np.column_stack(results)
    X_ind_norm = B @ C_ind

    # Unscale
    X_joint = X_joint_norm * sd + mu
    X_ind = X_ind_norm * sd + mu

    # Compute RMSE on missing values
    miss = ~mask
    rmse_joint = np.sqrt(np.mean((X_joint[miss, :] - X[miss, :]) ** 2))
    rmse_ind = np.sqrt(np.mean((X_ind[miss, :] - X[miss, :]) ** 2))

    return {
        "X_joint": X_joint,
        "X_ind": X_ind,
        "mask": mask,
        "rmse_joint": float(rmse_joint),
        "rmse_ind": float(rmse_ind),
        "mu": mu,
        "sd": sd,
    }
