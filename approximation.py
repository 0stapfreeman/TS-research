"""
Signal approximation functions using Fourier Ratio method.

This module implements both randomized and deterministic approaches for
approximating signals using trigonometric polynomials, as described in
Theorem 1.14 and 1.15 of the Talagrand constant paper.
"""

import numpy as np
from fourier_core import DFT_unitary, compute_required_fourier_terms


def build_Z(f: np.ndarray) -> np.ndarray | None:
    """
    Build a random Fourier component Z.

    IMPORTANT: This is the probabilistic construction from Theorem 1.14.
    It samples a frequency m WITH REPLACEMENT from the probability
    distribution P(m) = |f̂(m)| / ||f̂||₁.

    WHY SAMPLING WITH REPLACEMENT?
    - Each call to build_Z() samples ONE frequency (possibly repeating)
    - When k > N samples are drawn, some frequencies appear multiple times
    - This is a Monte Carlo method: variance decreases as 1/k
    - For practical use, Theorem 1.36 (large_coefficient_approx) is more efficient

    The "polynomial degree k" in Theorem 1.14 means NUMBER OF SAMPLES,
    not number of distinct frequencies. This can exceed N!

    Construction: Z = L1 × sgn(f̂(m)) × (1/√N) × χ_mx
    where χ_mx = e^(2πimx/N) is the character function.

    Parameters
    ----------
    f : np.ndarray
        Input signal of length N

    Returns
    -------
    np.ndarray or None
        Random Fourier component Z (complex array)
        Returns zeros if L1 norm is zero

    See Also
    --------
    approximate_f_by_Z : Averages k calls to build_Z
    large_coefficient_approx : Theorem 1.36 (deterministic, distinct frequencies only)
    """
    N = len(f)
    f_hat = DFT_unitary(f)

    abs_hat = np.abs(f_hat)
    L1 = abs_hat.sum()

    if L1 == 0:
        return np.zeros_like(f, dtype=complex)

    # Sample frequency m with probability |f̂(m)| / L1
    probs = abs_hat / L1
    m = np.random.choice(len(f_hat), p=probs)

    # Extract sign of f̂(m)
    if abs_hat[m] == 0:
        sgn = 0.0
    else:
        sgn = f_hat[m] / abs_hat[m]

    # Build character function χ_mx
    x_idx = np.arange(N)
    chi_mx = np.exp(2j * np.pi * m * x_idx / N)

    # Construct Z
    Z = L1 * sgn * (1 / np.sqrt(N)) * chi_mx

    return Z


def approximate_f_by_Z(f: np.ndarray, k: int) -> np.ndarray:
    """
    Approximate signal f by averaging k random Fourier components.

    This is the randomized approximation algorithm from Theorem 1.14.

    IMPORTANT CLARIFICATION:
    - Parameter k = NUMBER OF SAMPLES, not number of distinct frequencies
    - Each sample is drawn independently WITH REPLACEMENT
    - When k > N: Some frequencies appear multiple times in the average
    - This is a statistical method providing variance reduction

    WHY k CAN EXCEED N:
    Example for N=256, k=1202:
    - Sample frequency 1202 times with replacement
    - Frequency m₁ might be selected 5 times
    - Frequency m₂ might be selected 0 times
    - Frequency m₃ might be selected 3 times, etc.
    - Average all 1202 samples: P(x) = (1/1202)·Σᵢ₌₁¹²⁰² Zᵢ(x)

    FOR PRACTICAL USE: Consider using large_coefficient_approx() instead,
    which uses distinct frequencies only (Theorem 1.36).

    Parameters
    ----------
    f : np.ndarray
        Input signal of length N
    k : int
        Number of random samples to average (can exceed N!)

    Returns
    -------
    np.ndarray
        Approximation of f (complex array)

    See Also
    --------
    build_Z : Builds a single random Fourier component
    large_coefficient_approx : Theorem 1.36 (deterministic, more practical)
    deterministic_trig_approx : Derandomized version of this algorithm
    """
    N = len(f)
    acc = np.zeros(N, dtype=complex)
    for _ in range(k):
        Z = build_Z(f)
        acc += Z
    return acc / k


def deterministic_trig_approx(
    f: np.ndarray,
    eps: float = 1.0,
) -> tuple[np.ndarray, float, int, np.ndarray, np.ndarray]:
    """
    Deterministic trigonometric polynomial construction.

    This is the main approximation method based on derandomization of
    Theorem 1.15. It constructs a trigonometric polynomial P(x) that
    approximates f with relative error at most ε.

    The polynomial is: P(x) = Σ a[m] × e^(2πimx/N)

    where coefficients a[m] are determined by a derandomized sampling
    procedure that ensures deterministic approximation guarantees.

    Parameters
    ----------
    f : np.ndarray
        Input discrete signal of length N (complex or real)
    eps : float, optional
        Desired relative accuracy (η in the paper), default=1.0
        Controls the trade-off between accuracy and polynomial degree

    Returns
    -------
    P : np.ndarray
        Approximating trigonometric polynomial evaluated at all points
        (complex array of length N)
    rel_error : float
        Achieved relative L2 error: ||f - P||₂ / ||f||₂
    k : int
        Polynomial degree (number of terms used)
    a : np.ndarray
        Coefficients a[m] of the trigonometric polynomial
    c : np.ndarray
        Multiplicity counts c[m] used in derandomization
        (how many times each frequency was selected)
    """
    f = np.asarray(f, dtype=np.complex128)
    N = len(f)
    f_hat = DFT_unitary(f)
    abs_fhat = np.abs(f_hat)
    L1 = abs_fhat.sum()

    # Compute required number of terms k
    k = compute_required_fourier_terms(eps, f)

    # Derandomization: distribute k samples proportionally
    p = abs_fhat / L1  # Probability distribution
    t = k * p  # Expected counts
    c = np.floor(t).astype(int)  # Floor of expected counts

    # Distribute remaining samples to frequencies with largest fractional parts
    S = c.sum()
    if S < k:
        R = k - S
        frac = t - np.floor(t)
        idx = np.argsort(-frac)
        c[idx[:R]] += 1

    # Compute coefficients a[m]
    sign = np.where(f_hat == 0, 0, f_hat / abs_fhat)
    a = (L1 / (k * np.sqrt(N))) * sign * c

    # Evaluate polynomial P(x) = Σ a[m] × e^(2πimx/N)
    x = np.arange(N)
    m = np.arange(N)
    W = np.exp(2j * np.pi * np.outer(x, m) / N)
    P = W @ a

    # Compute relative error
    rel_error = np.linalg.norm(f - P) / np.linalg.norm(f)

    return P, rel_error, k, a, c


def large_coefficient_approx(f: np.ndarray, eta: float = 0.1) -> tuple:
    """
    Deterministic polynomial approximation using largest Fourier coefficients.

    Based on Theorem 1.36 from the paper. This method:
    - Uses ONLY DISTINCT frequencies (no repetition)
    - Selects frequencies where |f̂(m)| ≥ threshold
    - More efficient than randomized sampling for practical use

    The key difference from Theorem 1.14 (approximate_f_by_Z):
    - Theorem 1.14: Samples k frequencies WITH REPLACEMENT (k can exceed N)
    - Theorem 1.36: Keeps only distinct frequencies above threshold (≤ N)

    Parameters
    ----------
    f : np.ndarray
        Input signal of length N (real or complex)
    eta : float, optional
        Approximation accuracy parameter (default=0.1)
        Controls the threshold: smaller eta → more coefficients kept
        Guarantees relative error ≤ η

    Returns
    -------
    P : np.ndarray
        Approximating polynomial (complex array of length N)
    rel_error : float
        Achieved relative L2 error: ||f - P||₂ / ||f||₂
    k : int
        Number of distinct frequencies used (|Γ|)
    Gamma : np.ndarray
        Indices of selected large coefficients (the set Γ)

    Notes
    -----
    Theorem 1.36 states: For Γ = {m : |f̂(m)| ≥ η||f||₂/√N},
    the polynomial P(x) = (1/√N)Σ_{m∈Γ} f̂(m)e^(2πimx/N)
    satisfies ||f - P||₂ ≤ η||f||₂.

    The number of frequencies used is bounded by |Γ| ≤ FR(f)·√N/η,
    but in practice is often much smaller.

    Examples
    --------
    >>> f = np.random.randn(256)
    >>> P, err, k, Gamma = large_coefficient_approx(f, eta=0.1)
    >>> print(f"Used {k} distinct frequencies, error={err:.4f}")
    """
    f = np.asarray(f, dtype=np.complex128)
    N = len(f)
    f_hat = DFT_unitary(f)

    # Theorem 1.36: threshold = η·||f||₂ / √N
    threshold = eta * np.linalg.norm(f) / np.sqrt(N)

    # Find large coefficients: Γ = {m : |f̂(m)| ≥ threshold}
    large_mask = np.abs(f_hat) >= threshold
    Gamma = np.where(large_mask)[0]

    # Build polynomial using only large coefficients
    a = np.zeros_like(f_hat)
    a[Gamma] = f_hat[Gamma]

    # Reconstruct signal using inverse DFT
    # Use numpy's inverse FFT with proper normalization
    P = np.fft.ifft(np.fft.ifftshift(a)) * np.sqrt(N)

    # Compute relative error
    rel_error = np.linalg.norm(f - P) / np.linalg.norm(f)

    return P, rel_error, len(Gamma), Gamma


def large_coefficient_approx_adaptive(f: np.ndarray, eta: float = 0.1, percentile: float = 99.0):
    """
    Deterministic polynomial approximation using the largest Fourier coefficients,
    adapted to ignore practically irrelevant frequencies.

    Parameters
    ----------
    f : np.ndarray
        Input signal of length N (real or complex)
    eta : float
        Theoretical accuracy parameter (Theorem 1.36)
    percentile : float
        Practical threshold: keep only coefficients above this percentile

    Returns
    -------
    P : np.ndarray
        Approximating polynomial
    rel_error : float
        Achieved relative L2 error
    k : int
        Number of frequencies used
    Gamma : np.ndarray
        Indices of selected frequencies
    """
    f = np.asarray(f, dtype=np.complex128)
    N = len(f)

    # Compute unitary DFT
    f_hat = np.fft.fft(f) / np.sqrt(N)
    print(f"Computed DFT of length {N}")
    print(f"DFT magnitude stats: min={np.abs(f_hat).min():.6f}, median={np.median(np.abs(f_hat)):.6f}, mean={np.abs(f_hat).mean():.6f}, max={np.abs(f_hat).max():.6f}")

    # Theoretical threshold (Theorem 1.36)
    thresh_theory = eta * np.linalg.norm(f) / np.sqrt(N)
    print(f"Theoretical threshold (eta*||f||_2/sqrt(N)) = {thresh_theory:.6f}")

    # Practical threshold: keep only top percentile
    thresh_percentile = np.percentile(np.abs(f_hat), percentile)
    print(f"Practical threshold (top {percentile} percentile) = {thresh_percentile:.6f}")

    # Use the max of theoretical and practical threshold
    threshold = max(thresh_theory, thresh_percentile)
    print(f"Final threshold used = {threshold:.6f}")

    # Select frequencies above threshold
    Gamma = np.where(np.abs(f_hat) >= threshold)[0]
    print(f"Number of frequencies selected: {len(Gamma)}")
    print(f"Selected frequency indices (Gamma): {Gamma}")

    # Build approximation
    a = np.zeros_like(f_hat)
    a[Gamma] = f_hat[Gamma]
    P = np.fft.ifft(a) * np.sqrt(N)

    # Relative L2 error
    rel_error = np.linalg.norm(f - P) / np.linalg.norm(f)

    return P, rel_error, len(Gamma), Gamma


def periodic_forecast(a: np.ndarray, N: int, H: int) -> np.ndarray:
    """
    Forecast future values using trigonometric polynomial.

    Given the polynomial coefficients a[m], this function predicts
    values at future time points x = N, N+1, ..., N+H-1 by evaluating
    P(x) = Σ a[m] × e^(2πimx/N).

    This assumes the signal has periodic structure captured by the polynomial.

    Parameters
    ----------
    a : np.ndarray
        Trigonometric polynomial coefficients of length N
    N : int
        Original signal length (period of the polynomial)
    H : int
        Forecast horizon (number of future points to predict)

    Returns
    -------
    np.ndarray
        Predicted values for x = N, N+1, ..., N+H-1 (complex array)
    """
    m = np.arange(N)
    x_future = np.arange(N, N + H)

    # Evaluate polynomial at future points
    W_future = np.exp(2j * np.pi * np.outer(x_future, m) / N)
    P_future = W_future @ a

    return P_future
