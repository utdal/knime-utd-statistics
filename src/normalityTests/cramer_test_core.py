"""
Cramer-von Mises normality test computational core.

This module contains a theoretical implementation of the Cramer-von Mises
normality test, providing a similar interface to the Anderson-Darling test
for seamless integration into the unified normality testing node.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

# Import shared utilities from Anderson-Darling core
from .anderson_darling_core import (
    drop_nans_and_nonfinite,
    uniform_sample,
    sample_mean_std,
    zscore_inplace,
    stable_sort,
)


def _compute_cramer_statistic(sorted_z: np.ndarray) -> float:
    """
    Correct implementation of Cramer-von Mises statistic.

    Compares empirical CDF vs theoretical normal CDF.
    Formula: W² = (1/12n) + Σ[F_theoretical(z_i) - F_empirical(z_i)]²
    """
    n = sorted_z.size
    if n == 0:
        raise ValueError("No data.")

    # Theoretical CDF values (what we expect if data is normal)
    theoretical_cdf = norm.cdf(sorted_z)

    # Empirical CDF positions (what we actually observe)
    i = np.arange(1, n + 1)
    empirical_cdf = (i - 0.5) / n  # Midpoint formula commonly used

    # Cramer-von Mises statistic: measure squared differences
    W2 = (1.0 / (12 * n)) + np.sum((theoretical_cdf - empirical_cdf) ** 2)

    return float(W2)


def _cramer_critical_table() -> Tuple[np.ndarray, np.ndarray]:
    """
    Theoretical critical values for Cramer-von Mises test.

    In practice, you would use established tables or scipy implementations.
    These are simplified theoretical values for demonstration.
    """
    levels = np.array([0.15, 0.10, 0.05, 0.025, 0.01], dtype=float)
    crits = np.array(
        [0.091, 0.104, 0.126, 0.148, 0.178], dtype=float
    )  # Theoretical values
    return levels, crits


def _interpolate_cramer_critical(
    levels: np.ndarray, crits: np.ndarray, alpha: float
) -> Tuple[float, str]:
    """
    Interpolate critical value for Cramer-von Mises test.
    Similar logic to Anderson-Darling interpolation.
    """
    if alpha in levels:
        return float(crits[np.where(levels == alpha)[0][0]]), "exact"

    if alpha < levels.min():
        idx = np.argmin(levels)
        return float(crits[idx]), f"clamped_low({levels[idx]:.3g})"
    if alpha > levels.max():
        idx = np.argmax(levels)
        return float(crits[idx]), f"clamped_high({levels[idx]:.3g})"

    # Linear interpolation
    idx_hi = int(np.searchsorted(levels, alpha, side="left"))
    idx_lo = idx_hi - 1
    a0, a1 = levels[idx_lo], levels[idx_hi]
    c0, c1 = crits[idx_lo], crits[idx_hi]
    t = (alpha - a0) / (a1 - a0)
    c = c0 + t * (c1 - c0)
    return float(c), f"interp({a0:.3g}..{a1:.3g})"


def run_cramer_test(
    data: np.ndarray | pd.Series,
    *,
    alpha: float = 0.05,
    mu_sigma_mode: str = "estimate",
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    standardize: bool = False,
    sample_cap: Optional[int] = None,
    seed: int = 42,
    small_sample_correction: str = "auto",
) -> Dict[str, object]:
    """
    Run Cramer-von Mises Normality test.

    Note: This is a theoretical implementation. In practice, you would use
    scipy.stats.cramervonmises or similar established implementations.

    Parameters follow similar pattern to Anderson-Darling test for consistency.

    Returns a dict with 'summary', 'aux', and 'diagnostics' matching the
    Anderson-Darling interface for seamless integration.
    """
    s = pd.Series(data, copy=False).astype(float)
    n_raw = int(s.size)
    x = drop_nans_and_nonfinite(s.to_numpy())
    n_used_initial = int(x.size)
    n_dropped = n_raw - n_used_initial

    if x.size == 0:
        raise ValueError(
            "No valid observations after removing missing/non-finite values."
        )

    if np.allclose(np.std(x, ddof=1 if x.size > 1 else 0), 0.0, rtol=0, atol=1e-15):
        raise ValueError(
            "Input series is constant or near-constant; Cramer test undefined."
        )

    # Sampling
    sampling_applied = False
    if (
        sample_cap is not None
        and isinstance(sample_cap, int)
        and sample_cap > 0
        and x.size > sample_cap
    ):
        x = uniform_sample(x, sample_cap, seed=seed)
        sampling_applied = True

    # Parameter estimation
    if mu_sigma_mode not in ("estimate", "user"):
        raise ValueError("mu_sigma_mode must be 'estimate' or 'user'.")
    if mu_sigma_mode == "estimate":
        mu_used, sigma_used = sample_mean_std(x)
        if not np.isfinite(sigma_used) or sigma_used <= 0:
            raise ValueError(
                "Estimated standard deviation is non-positive; cannot proceed."
            )
    else:
        if mu is None or sigma is None or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(
                "When mu_sigma_mode='user', provide finite mu and sigma>0."
            )
        mu_used, sigma_used = float(mu), float(sigma)

    # Standardization
    standardized = False
    if standardize:
        x = zscore_inplace(x, mu_used, sigma_used)
        mu_used, sigma_used = 0.0, 1.0
        standardized = True

    # Standardize to N(0,1) for testing
    z = (x - mu_used) / sigma_used
    z = stable_sort(z)
    n_used = int(z.size)

    # Compute Cramer-von Mises statistic
    W2 = _compute_cramer_statistic(z)

    # Apply small sample correction if requested
    if small_sample_correction == "auto" and mu_sigma_mode == "estimate" and n_used > 0:
        # Standard correction for estimated parameters (Stephens, 1974)
        W2 *= 1.0 + 0.5 / n_used  # Corrected formula

    # Get critical value and make decision
    levels, crits = _cramer_critical_table()
    critical_value, note = _interpolate_cramer_critical(levels, crits, alpha)
    decision = "Reject normality" if W2 >= critical_value else "Do not reject normality"

    # Build diagnostics notes
    notes: List[str] = []
    if n_used < 8:
        notes.append("Small sample (n<8): low power / higher numerical sensitivity.")
    if n_used > 5000:
        notes.append("Very large sample: tiny deviations may be flagged (high power).")
    if standardized:
        notes.append("Data standardized (z-scored) prior to computation.")
    if sampling_applied:
        notes.append("Sampling cap applied for performance.")
    if mu_sigma_mode == "user":
        notes.append("Test performed with user-specified μ and σ.")
    if small_sample_correction == "auto" and mu_sigma_mode == "estimate":
        notes.append("Small-sample correction applied for estimated parameters.")

    # Prepare output in standardized format
    diagnostics = {
        "n_raw": n_raw,
        "n_used": n_used,
        "n_dropped": n_dropped,
        "sampling_applied": sampling_applied,
        "mu_sigma_mode": mu_sigma_mode,
        "mu_used": mu_used,
        "sigma_used": sigma_used,
        "standardized": standardized,
        "significance_mode": "critical",  # Cramer test typically uses critical values
        "critical_policy": "interpolate",
        "backend": "theoretical_cramer_implementation",
        "notes": notes,
    }

    summary = {
        "test": "Cramer-von Mises (Normal)",
        "n": n_used,
        "statistic_W2": W2,  # Different statistic name for clarity
        "p_value": None,  # Theoretical implementation doesn't compute p-values
        "alpha": alpha,
        "decision": decision,
        "critical_value": critical_value,
    }

    aux = {
        "critical_value_at_alpha": critical_value,
        "table_levels": levels.tolist(),
        "table_values": crits.tolist(),
        "alpha_mapping": note,
    }

    return {
        "summary": summary,
        "aux": aux,
        "diagnostics": diagnostics,
    }
