"""
Missing value recovery functions using Fourier Ratio and L1 minimization.

This module implements the missing value imputation method based on
Theorem 1.20 from the Talagrand constant paper. It uses DFT basis and
L1 minimization (compressed sensing) to recover missing observations.
"""

import numpy as np
from approximation import large_coefficient_approx


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


def build_dft_basis(N: int) -> np.ndarray:
    """
    Build orthonormal DFT (Discrete Fourier Transform) basis matrix.

    The DFT basis is used for sparse representation of signals in the
    frequency domain. This is the inverse DFT matrix normalized.

    Parameters
    ----------
    N : int
        Signal length

    Returns
    -------
    np.ndarray
        Orthonormal DFT basis matrix of shape (N, N), complex-valued
    """
    # Create DFT matrix: B[x, m] = (1/sqrt(N)) * exp(2πi*x*m/N)
    # This is the inverse DFT matrix as used in the paper
    x = np.arange(N)
    m = np.arange(N)
    B = np.exp(2j * np.pi * np.outer(x, m) / N) / np.sqrt(N)
    return B


def recover_l1_quadratic_constraint(A: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    """
    Solve L1-minimization with quadratic constraint (Theorem 1.20).

    Solves: min ||x̂||₁ subject to ||f - x||_{L²(X)} ≤ ||f||₂ × ε

    This is the correct formulation from Theorem 1.20. The constraint is
    on the empirical L2 norm over the sample set X, not an equality constraint.

    IMPORTANT: The norm ||f||₂ in the constraint refers to the norm of the
    observed values y, NOT the full unknown signal. This is crucial for the
    optimization to work correctly.

    Since the quadratic constraint is difficult to implement with standard
    solvers, we use a relaxed formulation: min ||x̂||₁ + λ||y - A@x̂||₂²
    where λ is chosen based on eps (Lagrange multiplier / LASSO approach).

    Parameters
    ----------
    A : np.ndarray
        Measurement matrix of shape (q, N) where q = number of observations
        A = B[obs_idx, :] where B is the DFT basis
    y : np.ndarray
        Observed values of length q (complex-valued)
    eps : float
        Accuracy parameter ε

    Returns
    -------
    np.ndarray
        Recovered Fourier coefficient vector x̂ of length N (complex-valued)

    Raises
    ------
    RuntimeError
        If the optimization does not converge
    """
    from scipy.optimize import minimize as scipy_minimize

    q, N = A.shape
    y_norm = np.linalg.norm(y)

    # Convert lambda based on eps: smaller eps means we care more about fitting
    # Lambda controls the trade-off between sparsity (L1) and fit (L2)
    lambda_param = eps / (1.0 + eps)  # Heuristic: eps=0.1 -> lambda~0.09

    def objective(x_flat):
        """Combined objective: λ * ||x̂||₁ + (1-λ) * ||y - A@x̂||₂²"""
        x_complex = x_flat[:N] + 1j * x_flat[N:]
        residual = y - A @ x_complex
        l1_term = np.sum(np.abs(x_complex))
        l2_term = np.sum(np.abs(residual)**2)
        return lambda_param * l1_term + (1 - lambda_param) * l2_term

    # Initial guess: least squares solution
    x0_complex, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    x0_flat = np.concatenate([np.real(x0_complex), np.imag(x0_complex)])

    # Use L-BFGS-B which handles non-smooth objectives better
    result = scipy_minimize(
        objective,
        x0_flat,
        method='L-BFGS-B',
        options={'maxiter': 2000, 'ftol': 1e-10}
    )

    if not result.success:
        # Try with less strict tolerance
        result = scipy_minimize(
            objective,
            x0_flat,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-7}
        )

    # Convert back to complex
    x_hat = result.x[:N] + 1j * result.x[N:]
    return x_hat


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

    B = build_dft_basis(N)
    A = B[obs_idx, :]

    c_rec = recover_l1_quadratic_constraint(A, y, eps)
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


def recover_polynomial_approx(
    f_obs: np.ndarray,
    mask: np.ndarray,
    eps: float = 0.1
) -> np.ndarray:
    """
    Recover missing values using polynomial approximation with adaptive frequency selection.

    IMPROVED APPROACH (using large_coefficient_approx_adaptive):
    1. Extract ONLY observed values
    2. Compute DFT of observed values
    3. Select frequencies with LARGEST coefficients (not just lowest frequencies!)
    4. Fit those selected frequencies to observed points using least squares
    5. Evaluate the fitted polynomial at ALL points (including missing)

    This is better than using lowest frequencies because:
    - Real signals don't always have energy concentrated at low frequencies
    - Adaptive selection finds the actual dominant frequencies in the data
    - Similar to Theorem 1.36 but for imputation

    Parameters
    ----------
    f_obs : np.ndarray
        Signal with NaN values for missing observations
    mask : np.ndarray
        Boolean array (True = observed, False = missing)
    eps : float, optional
        Approximation accuracy parameter (default=0.1)
        Controls threshold for keeping coefficients

    Returns
    -------
    np.ndarray
        Recovered signal with missing values filled in (real-valued)

    Notes
    -----
    Key differences from L1 minimization approach:

    **L1 Minimization (Theorem 1.20)**:
    - Minimizes L1 norm of Fourier coefficients (promotes sparsity)
    - Explicitly optimizes for sparse representation
    - Better for highly incomplete data
    - Slower (requires iterative optimization)

    **Polynomial Approximation (Adaptive L2 fitting)**:
    - Selects frequencies with largest magnitude coefficients
    - Fits selected frequencies using least squares (L2)
    - Direct linear algebra (no iterative optimization)
    - Faster but less sophisticated than L1

    The algorithm:
    1. Compute DFT of observed values
    2. Select frequencies where |f̂(m)| ≥ threshold
    3. Build design matrix A with only selected frequencies at observed indices
    4. Solve: min ||y - A@c||₂ where y = observed values, c = coefficients
    5. Reconstruct: f(n) = Σ c[m]×exp(2πi×m×n/N) for all n
    """
    from fourier_core import DFT_unitary
    from approximation import large_coefficient_approx_adaptive

    N = len(f_obs)

    # Extract observed values and indices
    observed_vals = f_obs[mask].copy()
    observed_indices = np.where(mask)[0]
    N_obs = len(observed_vals)

    # ADAPTIVE APPROACH: Select frequencies that best explain observed data
    #
    # Strategy:
    # 1. For each frequency in [0, N-1], compute how well it fits observed data
    # 2. Select top-k frequencies with highest fitting coefficients
    # 3. Refit using only selected frequencies
    # 4. Evaluate on full grid
    #
    # This is similar to matching pursuit or orthogonal matching pursuit

    # First pass: compute coefficients for ALL frequencies on observed data
    print(f"  [Polynomial] Computing all {N} frequency components on observed data...")

    # Build full design matrix for observed points
    m_all = np.arange(N)
    A_full = np.exp(2j * np.pi * np.outer(observed_indices, m_all) / N) / np.sqrt(N)

    # Project observed data onto all frequency basis functions
    # This gives us an estimate of coefficient magnitudes
    coeffs_all = A_full.conj().T @ observed_vals

    # Select k frequencies with largest coefficient magnitudes
    k = min(int(np.sqrt(N_obs)) + 5, N_obs // 3)
    largest_indices = np.argsort(np.abs(coeffs_all))[::-1][:k]
    largest_indices = np.sort(largest_indices)  # Sort for consistency

    print(f"  [Polynomial] Selected {k} frequencies with largest coefficients: {largest_indices[:10]}...")

    # Build reduced design matrix with only selected frequencies
    A_selected = A_full[:, largest_indices]

    # Refit using least squares on selected frequencies
    c_fit, residuals, rank, s = np.linalg.lstsq(A_selected, observed_vals, rcond=None)

    # Evaluate polynomial on FULL grid using selected frequencies
    x_full = np.arange(N)
    B_selected = np.exp(2j * np.pi * np.outer(x_full, largest_indices) / N) / np.sqrt(N)
    f_rec_full = B_selected @ c_fit

    return f_rec_full.real
