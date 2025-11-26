"""
Core mathematical functions for Fourier Ratio analysis.

This module implements the fundamental DFT and Fourier Ratio computations
based on the theoretical framework from the Talagrand constant paper.
"""

import numpy as np


def DFT_unitary(x: np.ndarray) -> np.ndarray:
    """
    Discrete Fourier Transform (unitary).

    Uses normalization factor 1/√N to preserve norms (Plancherel identity).
    This ensures that the unitary matrix preserves L2 norms.

    Formula: X̂(m) = (1/√N) × Σ e^(-2πixm/N) × x(i)

    Parameters
    ----------
    x : np.ndarray
        Input signal of length N

    Returns
    -------
    np.ndarray
        DFT coefficients (complex array of length N)
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    exp_ = np.exp(-1j * 2 * np.pi * k * n / N)
    X = (1 / np.sqrt(N)) * np.dot(exp_, x)
    return X


def fourier_ratio(x: np.ndarray) -> float:
    """
    Compute the Fourier Ratio (FR) of a signal.

    The Fourier Ratio is a complexity measure that relates to the learnability
    of a signal. It's defined as the ratio of L1 to L2 norms in Fourier domain.

    Formula: FR(f) = √N × (||f̂||_L1 / ||f̂||_L2) = (Σ|f̂(m)|) / √(Σ|f̂(m)|²)

    Interpretation:
    - FR ≈ 1: Low complexity, structured signal (easily learnable)
    - FR ≈ √N: High complexity, random signal (hard to learn)
    - Range: 1 ≤ FR(f) ≤ √N

    Parameters
    ----------
    x : np.ndarray
        Input signal of length N

    Returns
    -------
    float
        Fourier Ratio (real-valued scalar)
    """
    X = DFT_unitary(x)
    L1 = np.sum(np.abs(X))
    L2 = np.sqrt(np.sum(np.abs(X) ** 2))
    FR = L1 / L2
    return FR.real


def compute_required_fourier_terms(eta: float, x: np.ndarray) -> int:
    """
    Compute the required number of Fourier terms for approximation.

    Based on Theorem 1.14 from the paper: the number of terms needed to
    approximate a signal with relative accuracy η is approximately FR²/η².

    Formula: k = ⌊(FR² - 1)/η² + 1⌋

    Parameters
    ----------
    eta : float
        Desired approximation accuracy (relative error)
        Typically in range [0.1, 1.0]
    x : np.ndarray
        Input signal

    Returns
    -------
    int
        Number of Fourier terms (polynomial degree) required
    """
    FR = fourier_ratio(x)
    k = (FR**2 - 1) / (eta**2)
    return int(np.floor(k + 1))


def compute_large_coefficient_threshold(f: np.ndarray, eta: float) -> float:
    """
    Compute threshold for selecting large Fourier coefficients.

    Based on Theorem 1.36: τ = η·||f||₂ / √N

    This threshold determines which Fourier coefficients are considered "large enough"
    to be included in the approximation. Coefficients with |f̂(m)| ≥ τ are kept,
    while smaller coefficients are discarded.

    Parameters
    ----------
    f : np.ndarray
        Input signal of length N
    eta : float
        Accuracy parameter (controls approximation quality)
        Smaller eta → smaller threshold → more coefficients kept

    Returns
    -------
    float
        Threshold value τ

    Examples
    --------
    >>> f = np.random.randn(256)
    >>> threshold = compute_large_coefficient_threshold(f, eta=0.1)
    >>> # Keep all frequencies m where |f̂(m)| >= threshold
    """
    N = len(f)
    return eta * np.linalg.norm(f) / np.sqrt(N)
