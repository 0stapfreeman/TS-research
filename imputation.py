"""
Missing value recovery functions using Fourier Ratio and L1 minimization.

This module implements the missing value imputation method based on
Theorem 1.20 from the Talagrand constant paper. It uses DCT basis and
L1 minimization (compressed sensing) to recover missing observations.
"""

import numpy as np
from scipy import fftpack
from scipy.optimize import linprog


def mask_observations(f_full: np.ndarray, keep_prob: float, seed: int = 0):
    """
    Simulate missing data by randomly masking values.

    Randomly keeps each sample with probability keep_prob and sets
    the others to NaN to simulate missing observations.

    Parameters
    ----------
    f_full : np.ndarray
        Complete signal (no missing values)
    keep_prob : float
        Probability of observing each sample (1 - missing_rate)
        E.g., keep_prob=0.7 means 30% missing data
    seed : int, optional
        Random seed for reproducibility, default=0

    Returns
    -------
    mask : np.ndarray
        Boolean array (True = observed, False = missing)
    f_obs : np.ndarray
        Signal with NaN values for missing observations
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(len(f_full)) < keep_prob
    f_obs = f_full.copy()
    f_obs[~mask] = np.nan
    return mask, f_obs


def compute_q(FR: float, eps: float, N: int, C: float, max_available: int) -> int:
    """
    Compute number of observations needed for recovery.

    Based on Theorem 1.20: to recover a signal with Fourier Ratio FR
    and accuracy ε, we need at least q observations where:

    q = C × (FR²/ε²) × log²(FR/ε) × log(N)

    Parameters
    ----------
    FR : float
        Fourier Ratio of the signal
    eps : float
        Desired recovery accuracy (relative error)
    N : int
        Signal length
    C : float
        Universal constant multiplier (typically 1.0)
    max_available : int
        Maximum number of observations available
        (clips q to this value if needed)

    Returns
    -------
    int
        Number of observations to use for recovery
    """
    q_theor = int(C * FR**2 / eps**2 * np.log(FR / eps) ** 2 * np.log(N))
    return min(q_theor, max_available)


def build_dct_basis(N: int) -> np.ndarray:
    """
    Build orthonormal DCT (Discrete Cosine Transform) basis matrix.

    The DCT basis is used for sparse representation of signals in the
    frequency domain, similar to DFT but with real-valued coefficients.

    Parameters
    ----------
    N : int
        Signal length

    Returns
    -------
    np.ndarray
        Orthonormal DCT basis matrix of shape (N, N)
    """
    I = np.eye(N)
    B = fftpack.dct(I, type=2, norm="ortho", axis=0)
    return B


def recover_l1_via_lp(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve L1-minimization problem using linear programming.

    Solves: min ||c||₁ subject to Ac = y

    This is the core of compressed sensing / sparse recovery. We find
    the sparsest representation c (smallest L1 norm) that matches the
    observed values y through the measurement matrix A.

    The L1 minimization is converted to a linear program by introducing
    c = c_plus - c_minus with c_plus, c_minus ≥ 0, and minimizing
    ||c_plus||₁ + ||c_minus||₁.

    Parameters
    ----------
    A : np.ndarray
        Measurement matrix of shape (q, N) where q = number of observations
    y : np.ndarray
        Observed values of length q

    Returns
    -------
    np.ndarray
        Recovered coefficient vector c of length N

    Raises
    ------
    RuntimeError
        If the linear program does not converge
    """
    q, N = A.shape
    n_vars = N

    # Objective: minimize sum(c_plus + c_minus)
    c_obj = np.ones(2 * n_vars)

    # Equality constraint: A @ c_plus - A @ c_minus = y
    A_eq = np.hstack([A, -A])  # shape (q, 2N)
    b_eq = y

    # Bounds: c_plus >= 0, c_minus >= 0
    bounds = [(0, None)] * (2 * n_vars)

    # Solve LP using HiGHS method (modern, efficient solver)
    res = linprog(c=c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.status != 0:
        raise RuntimeError(f"LP did not converge: {res.message}")

    # Extract c = c_plus - c_minus
    c_plus = res.x[:n_vars]
    c_minus = res.x[n_vars:]
    c_rec = c_plus - c_minus
    return c_rec


def check_theorem_bound(f_full: np.ndarray, f_rec: np.ndarray, eps: float):
    """
    Verify Theorem 1.20 recovery bound.

    Theorem 1.20 states that with high probability:
    ||x* - f||₂ ≤ 11.47 × ||f||₂ × ε

    where x* is the recovered signal and f is the true signal.

    Parameters
    ----------
    f_full : np.ndarray
        True complete signal
    f_rec : np.ndarray
        Recovered signal
    eps : float
        Recovery accuracy parameter

    Returns
    -------
    lhs : float
        Left-hand side: ||x* - f||₂ (actual error)
    rhs : float
        Right-hand side: 11.47 × ||f||₂ × ε (theoretical bound)
    ratio : float
        lhs / rhs (should be ≤ 1 if bound holds)
    ok : bool
        True if bound holds (lhs ≤ rhs)
    """
    norm_f = np.linalg.norm(f_full)
    err = np.linalg.norm(f_rec - f_full)

    lhs = err
    rhs = 11.47 * norm_f * eps
    ok = lhs <= rhs

    return lhs, rhs, lhs / rhs, ok


def run_experiment(
    FR: float,
    sr: int = 32,
    seconds: float = 2.0,
    eps: float = 0.5,
    C: float = 1.0,
    keep_prob: float = 0.7,
    seed: int = 0,
):
    """
    Run complete missing value recovery experiment.

    This function orchestrates the entire imputation pipeline:
    1. Generates signal (requires signal_utils.sample_signal)
    2. Simulates missing data
    3. Computes required observations q
    4. Solves L1 minimization to recover missing values
    5. Verifies Theorem 1.20 bound
    6. Plots results (requires signal_utils.plot_reconstruction)

    Parameters
    ----------
    FR : float
        Fourier Ratio of the signal
    sr : int, optional
        Sampling rate in Hz, default=32
    seconds : float, optional
        Signal duration in seconds, default=2.0
    eps : float, optional
        Recovery accuracy parameter, default=0.5
    C : float, optional
        Universal constant multiplier, default=1.0
    keep_prob : float, optional
        Probability of observing each sample, default=0.7
    seed : int, optional
        Random seed for reproducibility, default=0

    Notes
    -----
    This function requires signal_utils module for signal generation
    and plotting. It's designed to be used in the demo notebook.
    """
    # NOTE: This function needs to be imported with signal_utils
    # It's included here for completeness but should be used in notebooks
    from signal_utils import sample_signal, plot_reconstruction

    print("Sampling rate SR:", sr)

    t, f_full = sample_signal(sr, seconds)
    N = len(f_full)

    print(f"FR: {FR:.4f}")

    mask, f_obs = mask_observations(f_full, keep_prob=keep_prob, seed=seed)
    valid_idx = np.where(mask)[0]

    q = compute_q(FR=FR, eps=eps, N=N, C=C, max_available=len(valid_idx))
    print(f"q (theoretical): {q}")

    obs_idx = np.random.choice(valid_idx, q, replace=False)
    y = f_obs[obs_idx]
    print(f"Number of used observations: {len(y)}")

    B = build_dct_basis(N)
    A = B[obs_idx, :]

    c_rec = recover_l1_via_lp(A, y)
    f_rec = B @ c_rec

    f_filled = f_obs.copy()
    f_filled[~mask] = f_rec[~mask]

    rel_err_full = np.linalg.norm(f_rec - f_full) / np.linalg.norm(f_full)
    rel_err_missing = np.linalg.norm(f_rec[~mask] - f_full[~mask]) / np.linalg.norm(
        f_full[~mask]
    )
    print(f"Relative error (full signal):    {rel_err_full:.4e}")
    print(f"Relative error (missing points): {rel_err_missing:.4e}")

    lhs, rhs, ratio, ok = check_theorem_bound(f_full, f_rec, eps)
    print("\n=== Theorem 1.20 bound check ===")
    print(f"||x* - f||_2        = {lhs:.4e}")
    print(f"11.47 ||f||_2 eps   = {rhs:.4e}")
    print(f"lhs / rhs           = {ratio:.3f}")
    print("✅ Bound holds." if ok else "❌ Bound does NOT hold for this eps, C, q.")

    plot_reconstruction(t, f_full, mask, f_obs, f_rec, seconds)
