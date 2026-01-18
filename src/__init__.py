# Fourier Ratio Time Series Research
from .fourier_core import DFT_unitary, fourier_ratio, compute_required_fourier_terms, compute_large_coefficient_threshold
from .approximation import build_Z, approximate_f_by_Z, large_coefficient_approx, deterministic_trig_approx, periodic_forecast
from .imputation import mask_observations, compute_q, build_dft_basis
from .joint_imputation import build_dct_basis, recover_joint, recover_independent, run_imputation_experiment
from .signal_utils import original_signal, sample_signal, plot_reconstruction, generate_composite_signal, plot_approximation_comparison, plot_forecast
