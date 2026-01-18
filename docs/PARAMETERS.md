# Fourier Ratio Research: Parameters Reference

This document provides a comprehensive reference for all parameters used in the Fourier Ratio research implementation.

---

## Table of Contents

1. [Core Fourier Ratio Parameters](#core-fourier-ratio-parameters)
2. [Signal Approximation Parameters](#signal-approximation-parameters)
3. [Missing Value Imputation Parameters](#missing-value-imputation-parameters)
4. [Signal Generation Parameters](#signal-generation-parameters)

---

## Core Fourier Ratio Parameters

### `N` - Signal Length
- **Type:** `int`
- **Description:** Number of samples in the discrete signal
- **Typical Values:** 64, 128, 256, 512, 1024
- **Notes:** Affects computational complexity and FR bounds (max FR = √N)

### `FR` - Fourier Ratio
- **Type:** `float`
- **Description:** Complexity measure of the signal
- **Formula:** `FR(f) = √N × (||f̂||_L1 / ||f̂||_L2)`
- **Range:** `[1, √N]`
- **Interpretation:**
  - `FR ≈ 1`: Low complexity, highly structured signal (easily learnable)
  - `FR ≈ √N`: High complexity, random signal (difficult to learn)
  - `FR < √N/2`: Generally considered "small" FR (good for approximation/recovery)
- **Notes:** Computed automatically from signal using `fourier_ratio(x)`

---

## Signal Approximation Parameters

### `eps` / `eta` / `ε` - Approximation Accuracy
- **Type:** `float`
- **Description:** Desired relative accuracy for polynomial approximation
- **Formula:** Controls target error `||f - P||₂ / ||f||₂ ≤ ε`
- **Typical Values:** `0.05` to `1.0`
  - `0.05`: Very high accuracy (requires many terms)
  - `0.1`: High accuracy (standard choice)
  - `0.5`: Moderate accuracy
  - `1.0`: Low accuracy (fewer terms, faster)
- **Trade-off:** Smaller ε → higher accuracy → larger polynomial degree k

### `k` - Polynomial Degree
- **Type:** `int`
- **Description:** Number of Fourier terms (frequencies) used in approximation
- **Formula:** `k = ⌊(FR² - 1)/ε² + 1⌋`
- **Computed From:** `FR` and `ε` using `compute_required_fourier_terms(eta, x)`
- **Typical Range:** 10 to 1000 (depends on FR and ε)
- **Notes:**
  - Directly affects computational cost
  - Higher k = better approximation but slower
  - Scales quadratically with FR: `k ∝ FR²/ε²`

### `a` - Polynomial Coefficients
- **Type:** `np.ndarray` (complex, length N)
- **Description:** Coefficients of the trigonometric polynomial `P(x) = Σ a[m] × e^(2πimx/N)`
- **Returned By:** `deterministic_trig_approx()`
- **Usage:** Used for evaluating polynomial and forecasting

### `c` - Multiplicity Counts
- **Type:** `np.ndarray` (int, length N)
- **Description:** Number of times each frequency m was selected in derandomization
- **Returned By:** `deterministic_trig_approx()`
- **Notes:** `sum(c) = k` (total polynomial degree)

### `H` - Forecast Horizon
- **Type:** `int`
- **Description:** Number of future time points to predict
- **Typical Values:** 10 to 500
- **Usage:** `periodic_forecast(a, N, H)` predicts points `x = N, N+1, ..., N+H-1`
- **Assumptions:** Signal has periodic structure captured by the polynomial

---

## Missing Value Imputation Parameters

### `keep_prob` - Observation Probability
- **Type:** `float`
- **Description:** Probability that each sample is observed (not missing)
- **Range:** `(0, 1]`
- **Typical Values:**
  - `0.9`: 10% missing data
  - `0.8`: 20% missing data
  - `0.7`: 30% missing data (common test case)
  - `0.6`: 40% missing data
  - `0.5`: 50% missing data (challenging)
- **Inverse:** `missing_rate = 1 - keep_prob`
- **Notes:** Lower values = more missing data = harder recovery

### `q` - Required Observations
- **Type:** `int`
- **Description:** Number of observations needed for successful recovery
- **Formula:** `q = C × (FR²/ε²) × log²(FR/ε) × log(N)`
- **Computed By:** `compute_q(FR, eps, N, C, max_available)`
- **Based On:** Theorem 1.20 from the paper
- **Notes:**
  - Must satisfy `q ≤ number of available observations`
  - Scales with signal complexity (FR) and desired accuracy (ε)

### `eps` / `ε` - Recovery Accuracy Parameter
- **Type:** `float`
- **Description:** Target recovery accuracy for imputation
- **Typical Values:** `0.1` to `0.5`
  - `0.1`: High accuracy recovery (requires more observations)
  - `0.2`: Moderate accuracy
  - `0.5`: Lower accuracy (fewer observations needed)
- **Theorem Bound:** `||x* - f||₂ ≤ 11.47 × ||f||₂ × ε`
- **Trade-off:** Smaller ε → better recovery → more observations needed (q)

### `C` - Universal Constant Multiplier
- **Type:** `float`
- **Description:** Constant multiplier in observation count formula
- **Default:** `1.0`
- **Typical Range:** `0.5` to `2.0`
- **Usage:** Can be adjusted to:
  - Increase (`C > 1`): Use more observations for more robust recovery
  - Decrease (`C < 1`): Use fewer observations (risk lower accuracy)
- **Notes:** Theoretical value is C = 1; practical values may vary

### `seed` - Random Seed
- **Type:** `int`
- **Description:** Random number generator seed for reproducibility
- **Typical Values:** `0`, `42`, or any integer
- **Usage:**
  - Ensures reproducible missing data patterns
  - Ensures reproducible observation selection
- **Notes:** Set for experiments; omit for production randomness

### `mask` - Observation Mask
- **Type:** `np.ndarray` (boolean, length N)
- **Description:** Boolean array indicating observed vs missing values
- **Values:**
  - `True`: Sample is observed (available)
  - `False`: Sample is missing (needs recovery)
- **Returned By:** `mask_observations(f_full, keep_prob, seed)`

### Recovery Matrices

#### `B` - DCT Basis Matrix
- **Type:** `np.ndarray` (float, shape N × N)
- **Description:** Orthonormal Discrete Cosine Transform basis
- **Created By:** `build_dct_basis(N)`
- **Usage:** Sparse representation of signals
- **Properties:** Orthonormal (B^T @ B = I)

#### `A` - Measurement Matrix
- **Type:** `np.ndarray` (float, shape q × N)
- **Description:** Submatrix of B corresponding to observed indices
- **Construction:** `A = B[obs_idx, :]`
- **Usage:** Used in L1 minimization constraint `Ac = y`

#### `y` - Observed Values
- **Type:** `np.ndarray` (float, length q)
- **Description:** Vector of observed signal values
- **Construction:** `y = f_obs[obs_idx]`
- **Usage:** Right-hand side of recovery equation

---

## Signal Generation Parameters

### `sr` - Sampling Rate
- **Type:** `int`
- **Description:** Number of samples per second (Hz)
- **Typical Values:**
  - `32 Hz`: Low rate (faster computation)
  - `256 Hz`: Standard rate
  - `1000 Hz`: High rate (more detail)
- **Units:** Hertz (Hz) or samples/second
- **Notes:** Total samples `N = sr × seconds`

### `seconds` - Signal Duration
- **Type:** `float`
- **Description:** Duration of the signal in seconds
- **Typical Values:** `1.0` to `10.0`
- **Units:** Seconds
- **Notes:** Longer signals = more samples = higher computational cost

### `t` - Time Grid
- **Type:** `np.ndarray` (float, length N)
- **Description:** Time points at which signal is sampled
- **Generation:** `t = np.linspace(0, seconds, N, endpoint=False)`
- **Units:** Seconds
- **Usage:** X-axis for plotting and signal evaluation

### Composite Signal Parameters

#### `frequencies` - Frequency Components
- **Type:** `list[tuple[float, float, float]]`
- **Description:** List of `(amplitude, frequency_hz, phase)` tuples for composite signals
- **Format:** Each tuple defines one sine wave component
  - `amplitude`: Scaling factor (0 to 1 typically)
  - `frequency_hz`: Frequency in Hz
  - `phase`: Phase offset in radians (0 to 2π)
- **Example:** `[(1.0, 2, 0), (0.7, 5, 0.5)]` creates signal with two sine waves
- **Usage:** `generate_composite_signal(sr, seconds, frequencies)`

---

## Parameter Relationships and Formulas

### Approximation

```
k = ⌊(FR² - 1)/ε² + 1⌋
Approximation error ≤ ε
Computational cost ∝ k × N
```

### Imputation

```
q = C × (FR²/ε²) × log²(FR/ε) × log(N)
Recovery bound: ||x* - f||₂ ≤ 11.47 × ||f||₂ × ε
Success requires: q ≤ (available observations)
```

### Signal Generation

```
N = sr × seconds
t = [0, Δt, 2Δt, ..., (N-1)Δt]  where Δt = 1/sr
```

---

## Quick Reference Table

| Parameter | Type | Typical Range | Purpose | Module |
|-----------|------|---------------|---------|--------|
| `N` | int | 64-1024 | Signal length | All |
| `FR` | float | 1-√N | Complexity measure | Core |
| `eps`/`ε` | float | 0.05-1.0 | Accuracy target | Approx, Impute |
| `k` | int | 10-1000 | Polynomial degree | Approx |
| `H` | int | 10-500 | Forecast horizon | Approx |
| `keep_prob` | float | 0.5-0.9 | Observation rate | Impute |
| `q` | int | 10-N | Observations needed | Impute |
| `C` | float | 0.5-2.0 | Constant multiplier | Impute |
| `sr` | int | 32-1000 | Sampling rate (Hz) | Signal |
| `seconds` | float | 1.0-10.0 | Duration (s) | Signal |
| `seed` | int | any | Random seed | Impute |

---

## Parameter Selection Guidelines

### For Signal Approximation:
1. **Start with:** `eps = 0.1`, compute `k` automatically
2. **For faster computation:** Increase `eps` (e.g., `0.5`)
3. **For higher accuracy:** Decrease `eps` (e.g., `0.05`)
4. **Forecast horizon:** Choose `H` based on application needs

### For Missing Value Imputation:
1. **Low missing rate (<20%):** `eps = 0.1`, `C = 1.0`
2. **Moderate missing rate (20-40%):** `eps = 0.2`, `C = 1.5`
3. **High missing rate (>40%):** `eps = 0.3-0.5`, `C = 2.0`
4. **Check:** Ensure `q ≤ available_observations`
5. **Verify:** Check if Theorem 1.20 bound holds after recovery

### General:
- Signals with **small FR** (< 5): Easy to approximate and recover
- Signals with **large FR** (> 10): Difficult, may need more observations or relaxed accuracy
- Always verify results with error metrics and visualizations
