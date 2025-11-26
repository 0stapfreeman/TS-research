"""
Utility functions for signal generation and visualization.

This module provides helper functions for creating test signals and
plotting results for the Fourier Ratio research demos.
"""

import numpy as np
import matplotlib.pyplot as plt


def original_signal(x: np.ndarray) -> np.ndarray:
    """
    Generate a test signal composed of two sine waves.

    This is a simple periodic signal with low Fourier Ratio (structured).
    Formula: sin(2π×2×x) + 0.7×sin(2π×5×x + 0.5)

    Parameters
    ----------
    x : np.ndarray
        Time points (typically normalized to [0, 1])

    Returns
    -------
    np.ndarray
        Signal values at given time points
    """
    return np.sin(2 * np.pi * 2 * x) + 0.7 * np.sin(2 * np.pi * 5 * x + 0.5)


def sample_signal(sr: int, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate discretized version of the test signal.

    Creates a time grid and samples the original_signal at those points.

    Parameters
    ----------
    sr : int
        Sampling rate in Hz (samples per second)
    seconds : float
        Duration of the signal in seconds

    Returns
    -------
    t : np.ndarray
        Time grid of length N = sr × seconds
    f_full : np.ndarray
        Discretized signal values at time points t
    """
    N = int(sr * seconds)
    t = np.linspace(0, seconds, N, endpoint=False)
    f_full = original_signal(t)
    return t, f_full


def plot_reconstruction(
    t: np.ndarray,
    f_full: np.ndarray,
    mask: np.ndarray,
    f_obs: np.ndarray,
    f_rec: np.ndarray,
    seconds: float,
):
    """
    Plot true, observed, and reconstructed signals for imputation demo.

    Creates a visualization showing:
    - True signal (dense and discrete versions)
    - Observed points (non-missing values)
    - Reconstructed signal from L1 minimization

    Parameters
    ----------
    t : np.ndarray
        Time grid for discrete signal
    f_full : np.ndarray
        True complete signal (discrete)
    mask : np.ndarray
        Boolean array (True = observed, False = missing)
    f_obs : np.ndarray
        Signal with NaN for missing values
    f_rec : np.ndarray
        Reconstructed signal from recovery algorithm
    seconds : float
        Signal duration (for plotting dense version)
    """
    # Create dense version for smooth visualization
    t_dense = np.linspace(0, seconds, 256, endpoint=False)
    f_dense = original_signal(t_dense)

    plt.figure(figsize=(10, 4))
    plt.plot(t_dense, f_dense, "y", label="True signal (dense)")
    plt.plot(t, f_full, "k", alpha=0.6, label="True signal (discrete)")
    plt.scatter(t[mask], f_obs[mask], c="b", s=20, label="Observed")
    plt.plot(t, f_rec, "r--", label="Reconstructed (DCT + L1)")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.title("Recovery of missing values via DCT + L1 (scipy.linprog)")
    plt.show()


def generate_composite_signal(
    sr: int, seconds: float, frequencies: list[tuple[float, float, float]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a composite signal from multiple sine waves.

    This allows creating more complex test signals with multiple frequency
    components for testing the approximation and imputation methods.

    Parameters
    ----------
    sr : int
        Sampling rate in Hz
    seconds : float
        Duration in seconds
    frequencies : list of tuples, optional
        List of (amplitude, frequency_hz, phase) tuples
        If None, uses default composite signal

    Returns
    -------
    t : np.ndarray
        Time grid
    x : np.ndarray
        Composite signal

    Examples
    --------
    >>> t, x = generate_composite_signal(256, 1, [(1.0, 2, 0), (0.7, 5, 0.5)])
    """
    N = int(sr * seconds)
    t = np.linspace(0, seconds, N, endpoint=False)

    if frequencies is None:
        # Default: complex composite signal with frequency modulation
        x = (
            np.sin(2 * np.pi * 2 * t)
            + 0.7 * np.sin(2 * np.pi * 5 * t + 0.5)
            + 0.5 * np.sin(2 * np.pi * (10 + 3 * np.sin(2 * np.pi * 0.2 * t)) * t)
        )
    else:
        # Custom composite from frequency list
        x = np.zeros(N)
        for amp, freq, phase in frequencies:
            x += amp * np.sin(2 * np.pi * freq * t + phase)

    return t, x


def plot_approximation_comparison(
    t: np.ndarray,
    f_true: np.ndarray,
    f_approx: np.ndarray,
    title: str = "True vs Approximated Signal",
):
    """
    Plot comparison between true and approximated signals.

    Parameters
    ----------
    t : np.ndarray
        Time grid
    f_true : np.ndarray
        True signal
    f_approx : np.ndarray
        Approximated signal
    title : str, optional
        Plot title
    """
    plt.figure(figsize=(10, 5))
    plt.plot(t, f_true.real, color="blue", label="Original signal f(x)")
    plt.plot(
        t, f_approx.real, color="orange", label="Approximation P(x)", alpha=0.7, linestyle="--"
    )
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_forecast(
    t: np.ndarray,
    f_true: np.ndarray,
    f_pred: np.ndarray,
    f_future: np.ndarray,
    N: int,
    H: int,
):
    """
    Plot true signal, predicted values, and forecast.

    Parameters
    ----------
    t : np.ndarray
        Time grid for true signal (length N)
    f_true : np.ndarray
        True signal values
    f_pred : np.ndarray
        Predicted/approximated values for observed region
    f_future : np.ndarray
        Forecasted future values (length H)
    N : int
        Original signal length
    H : int
        Forecast horizon
    """
    t_full = np.arange(N + H)

    plt.figure(figsize=(12, 6))
    plt.plot(t_full[:N], f_true.real, label="True signal f(x)")
    plt.plot(t_full[:N], f_pred.real, label="Predicted P(x)", linestyle="--")
    plt.plot(t_full[N:], f_future.real, label="Future forecast P(x+N)", linestyle="--")
    plt.axvline(x=N, color="gray", linestyle=":", alpha=0.5, label="Forecast boundary")
    plt.title("True Signal, Deterministic Approximation, and Periodic Forecast")
    plt.xlabel("Time index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
