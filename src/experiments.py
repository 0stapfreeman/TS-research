"""
Experiment utilities for statistical testing with multiple seeds.

Provides functions to run experiments multiple times and compute
statistical significance of results.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .joint_imputation import run_imputation_experiment, build_dct_basis, recover_joint, recover_independent
from .baselines import impute_mean, impute_linear, impute_knn
from joblib import Parallel, delayed


def create_missing_mask(
    N: int,
    missing_type: str = "random",
    keep_prob: float = 0.7,
    block_size: int = 10,
    num_blocks: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """
    Create a boolean mask for observed values with different missing patterns.

    Parameters
    ----------
    N : int
        Number of timesteps
    missing_type : str
        Type of missingness:
        - "random": Random point-wise missingness (Bernoulli)
        - "block": Contiguous blocks of missing values
        - "burst": Multiple random bursts of missing values
    keep_prob : float
        For "random": probability of keeping each point
        For "block"/"burst": approximate fraction to keep (adjusted by block sizes)
    block_size : int
        For "block"/"burst": size of each missing block
    num_blocks : int
        For "burst": number of missing blocks (ignored for "block")
    seed : int
        Random seed

    Returns
    -------
    np.ndarray
        Boolean mask of shape (N,), True = observed
    """
    rng = np.random.default_rng(seed)

    if missing_type == "random":
        mask = rng.random(N) < keep_prob

    elif missing_type == "block":
        # Single contiguous block of missing values in the middle
        mask = np.ones(N, dtype=bool)
        n_missing = int(N * (1 - keep_prob))
        start = (N - n_missing) // 2
        mask[start : start + n_missing] = False

    elif missing_type == "burst":
        # Multiple random bursts of missing values
        mask = np.ones(N, dtype=bool)
        # Calculate number of blocks to achieve approximate keep_prob
        total_missing = int(N * (1 - keep_prob))
        actual_num_blocks = max(1, total_missing // block_size)

        # Randomly place blocks (non-overlapping)
        available_starts = list(range(0, N - block_size + 1))
        rng.shuffle(available_starts)

        blocks_placed = 0
        for start in available_starts:
            if blocks_placed >= actual_num_blocks:
                break
            # Check if this block overlaps with existing missing
            if mask[start : start + block_size].all():
                mask[start : start + block_size] = False
                blocks_placed += 1

    else:
        raise ValueError(f"Unknown missing_type: {missing_type}")

    return mask


def run_experiment_with_mask(
    X: np.ndarray,
    mask: np.ndarray,
    delta_rel: float = 0.01,
    solver: str = "ECOS",
    knn_k: int = 5,
) -> dict:
    """
    Run imputation experiment with a pre-specified mask.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, m)
    mask : np.ndarray
        Boolean mask of shape (N,), True = observed
    delta_rel : float
        Relative noise tolerance for sparse recovery
    solver : str
        CVXPY solver to use
    knn_k : int
        Number of neighbors for KNN imputation

    Returns
    -------
    dict with RMSE for each method and recovered matrices
    """
    N, m = X.shape
    miss = ~mask
    obs_idx = np.where(mask)[0]

    # Standardize per series
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    X_norm = (X - mu) / sd

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

    results_ind = Parallel(n_jobs=-1)(delayed(solve_one)(j) for j in range(m))
    C_ind = np.column_stack(results_ind)
    X_ind_norm = B @ C_ind

    # Unscale
    X_joint = X_joint_norm * sd + mu
    X_ind = X_ind_norm * sd + mu

    # Run baseline methods
    X_mean = impute_mean(X, mask)
    X_linear = impute_linear(X, mask)
    X_knn = impute_knn(X, mask, k=knn_k)

    # Compute RMSE on missing values
    rmse_joint = np.sqrt(np.mean((X_joint[miss, :] - X[miss, :]) ** 2))
    rmse_ind = np.sqrt(np.mean((X_ind[miss, :] - X[miss, :]) ** 2))
    rmse_mean = np.sqrt(np.mean((X_mean[miss, :] - X[miss, :]) ** 2))
    rmse_linear = np.sqrt(np.mean((X_linear[miss, :] - X[miss, :]) ** 2))
    rmse_knn = np.sqrt(np.mean((X_knn[miss, :] - X[miss, :]) ** 2))

    return {
        "rmse_joint": float(rmse_joint),
        "rmse_ind": float(rmse_ind),
        "rmse_mean": float(rmse_mean),
        "rmse_linear": float(rmse_linear),
        "rmse_knn": float(rmse_knn),
        "X_joint": X_joint,
        "X_ind": X_ind,
        "X_linear": X_linear,
        "mask": mask,
    }


def run_multi_seed_experiment_with_pattern(
    X: np.ndarray,
    missing_type: str = "random",
    keep_prob: float = 0.7,
    block_size: int = 10,
    delta_rel: float = 0.01,
    seeds: list[int] | None = None,
    solver: str = "ECOS",
    knn_k: int = 5,
) -> pd.DataFrame:
    """
    Run imputation experiment with multiple seeds and configurable missing pattern.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, m)
    missing_type : str
        Type of missingness: "random", "block", or "burst"
    keep_prob : float
        Fraction of timesteps to keep as observed
    block_size : int
        Size of missing blocks (for "burst" pattern)
    delta_rel : float
        Relative noise tolerance
    seeds : list of int
        List of random seeds (default: [0, 1, 2, ..., 9])
    solver : str
        CVXPY solver to use
    knn_k : int
        Number of neighbors for KNN

    Returns
    -------
    pd.DataFrame with columns:
        seed, rmse_joint, rmse_ind, rmse_mean, rmse_linear, rmse_knn
    """
    if seeds is None:
        seeds = list(range(10))

    N = X.shape[0]
    results = []

    for seed in seeds:
        mask = create_missing_mask(
            N,
            missing_type=missing_type,
            keep_prob=keep_prob,
            block_size=block_size,
            seed=seed,
        )
        result = run_experiment_with_mask(
            X, mask, delta_rel=delta_rel, solver=solver, knn_k=knn_k
        )
        result["seed"] = seed
        # Remove large arrays from result
        result = {k: v for k, v in result.items() if k not in ["X_joint", "X_ind", "mask"]}
        results.append(result)

    return pd.DataFrame(results)


def run_single_experiment_with_baselines(
    X: np.ndarray,
    keep_prob: float = 0.7,
    delta_rel: float = 0.01,
    seed: int = 0,
    solver: str = "ECOS",
    knn_k: int = 5,
) -> dict:
    """
    Run imputation experiment with all methods (joint, independent, baselines).

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, m)
    keep_prob : float
        Fraction of timesteps to keep as observed
    delta_rel : float
        Relative noise tolerance for sparse recovery
    seed : int
        Random seed
    solver : str
        CVXPY solver to use
    knn_k : int
        Number of neighbors for KNN imputation

    Returns
    -------
    dict with RMSE for each method
    """
    # Run the main experiment (joint and independent)
    result = run_imputation_experiment(
        X, keep_prob=keep_prob, delta_rel=delta_rel, seed=seed, solver=solver
    )

    mask = result["mask"]
    miss = ~mask

    # Run baseline methods
    X_mean = impute_mean(X, mask)
    X_linear = impute_linear(X, mask)
    X_knn = impute_knn(X, mask, k=knn_k)

    # Compute RMSE for baselines
    rmse_mean = np.sqrt(np.mean((X_mean[miss, :] - X[miss, :]) ** 2))
    rmse_linear = np.sqrt(np.mean((X_linear[miss, :] - X[miss, :]) ** 2))
    rmse_knn = np.sqrt(np.mean((X_knn[miss, :] - X[miss, :]) ** 2))

    return {
        "seed": seed,
        "rmse_joint": result["rmse_joint"],
        "rmse_ind": result["rmse_ind"],
        "rmse_mean": float(rmse_mean),
        "rmse_linear": float(rmse_linear),
        "rmse_knn": float(rmse_knn),
    }


def run_multi_seed_experiment(
    X: np.ndarray,
    keep_prob: float = 0.7,
    delta_rel: float = 0.01,
    seeds: list[int] | None = None,
    solver: str = "ECOS",
    knn_k: int = 5,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Run imputation experiment with multiple seeds.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, m)
    keep_prob : float
        Fraction of timesteps to keep as observed
    delta_rel : float
        Relative noise tolerance
    seeds : list of int
        List of random seeds (default: [0, 1, 2, ..., 9])
    solver : str
        CVXPY solver to use
    knn_k : int
        Number of neighbors for KNN

    Returns
    -------
    pd.DataFrame with columns:
        seed, rmse_joint, rmse_ind, rmse_mean, rmse_linear, rmse_knn
    """
    if seeds is None:
        seeds = list(range(10))

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_single_experiment_with_baselines)(
            X,
            keep_prob=keep_prob,
            delta_rel=delta_rel,
            seed=seed,
            solver=solver,
            knn_k=knn_k,
        )
        for seed in seeds
    )

    return pd.DataFrame(results)


def compute_statistics(results_df: pd.DataFrame) -> dict:
    """
    Compute mean, std, and significance tests for experiment results.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame from run_multi_seed_experiment

    Returns
    -------
    dict with:
        - mean_<method>: mean RMSE for each method
        - std_<method>: standard deviation for each method
        - pvalue_joint_vs_<method>: paired t-test p-value (joint vs each method)
        - wilcoxon_joint_vs_<method>: Wilcoxon signed-rank p-value
    """
    methods = ["joint", "ind", "mean", "linear", "knn"]
    stats_dict = {}

    # Compute mean and std for each method
    for method in methods:
        col = f"rmse_{method}"
        stats_dict[f"mean_{method}"] = results_df[col].mean()
        stats_dict[f"std_{method}"] = results_df[col].std()

    # Paired t-tests (joint vs each other method)
    joint_rmse = results_df["rmse_joint"].values
    for method in methods[1:]:  # Skip joint vs joint
        other_rmse = results_df[f"rmse_{method}"].values
        # Paired t-test (two-sided)
        t_stat, p_value = stats.ttest_rel(joint_rmse, other_rmse)
        stats_dict[f"pvalue_joint_vs_{method}"] = p_value

        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            w_stat, w_pvalue = stats.wilcoxon(joint_rmse, other_rmse)
            stats_dict[f"wilcoxon_joint_vs_{method}"] = w_pvalue
        except ValueError:
            # Can fail if differences are all zero
            stats_dict[f"wilcoxon_joint_vs_{method}"] = np.nan

    return stats_dict


def format_results_table(stats: dict, dataset_name: str = "") -> str:
    """
    Format results as a LaTeX table for paper.

    Parameters
    ----------
    stats : dict
        Statistics dictionary from compute_statistics
    dataset_name : str
        Name of the dataset for the table caption

    Returns
    -------
    str
        LaTeX table code
    """
    methods = [
        ("Joint (ℓ2,1)", "joint"),
        ("Independent (ℓ1)", "ind"),
        ("Mean", "mean"),
        ("Linear", "linear"),
        ("KNN", "knn"),
    ]

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{Imputation Results{': ' + dataset_name if dataset_name else ''}}}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Method & RMSE & p-value (vs Joint) \\\\",
        "\\midrule",
    ]

    for display_name, key in methods:
        mean = stats[f"mean_{key}"]
        std = stats[f"std_{key}"]
        if key == "joint":
            pval_str = "-"
        else:
            pval = stats.get(f"pvalue_joint_vs_{key}", np.nan)
            if np.isnan(pval):
                pval_str = "-"
            elif pval < 0.001:
                pval_str = "$<$0.001"
            else:
                pval_str = f"{pval:.3f}"

        lines.append(f"{display_name} & {mean:.4f} $\\pm$ {std:.4f} & {pval_str} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)
