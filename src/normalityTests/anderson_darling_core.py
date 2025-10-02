"""
Anderson-Darling normality test computational core.

This module contains the pure Python implementation of the Anderson-Darling
normality test, separated from KNIME UI logic for better maintainability
and testability.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

# Optional: statsmodels for AD p-value
try:
    from statsmodels.stats.diagnostic import normal_ad as _sm_normal_ad

    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


def drop_nans_and_nonfinite(x: np.ndarray) -> np.ndarray:
    """Remove NaN and +/-Inf; return cleaned 1-D float array."""
    x = np.asarray(x, dtype=float).ravel()
    mask = np.isfinite(x)
    return x[mask]


def uniform_sample(x: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Deterministic uniform sample without replacement."""
    if k is None or k <= 0 or x.size <= k:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=k, replace=False)
    return x[np.sort(idx)]


def sample_mean_std(x: np.ndarray) -> Tuple[float, float]:
    """Return sample mean and (unbiased) sample std (ddof=1 if n>1 else ddof=0)."""
    n = x.size
    mu = float(np.mean(x)) if n else float("nan")
    ddof = 1 if n > 1 else 0
    sigma = float(np.std(x, ddof=ddof)) if n else float("nan")
    return mu, sigma


def zscore_inplace(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Return standardized copy (x - mu)/sigma. Caller must ensure sigma>0."""
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("Standard deviation must be positive to standardize.")
    return (x - mu) / sigma


def stable_sort(x: np.ndarray) -> np.ndarray:
    """Return a sorted copy using a stable O(n log n) comparison sort."""
    return np.sort(x, kind="mergesort")


def _ad_critical_table_for_normal_estimated() -> Tuple[np.ndarray, np.ndarray]:
    """
    Anderson–Darling critical values for Normal when mu,sigma are estimated from data.
    Levels and values match common tables (e.g., SciPy reference):
      sig levels: 15%, 10%, 5%, 2.5%, 1%
    """
    levels = np.array([0.15, 0.10, 0.05, 0.025, 0.01], dtype=float)
    crits = np.array([0.576, 0.656, 0.787, 0.918, 1.092], dtype=float)
    return levels, crits


def _interpolate_critical_value(
    levels: np.ndarray, crits: np.ndarray, alpha: float, policy: str = "interpolate"
) -> Tuple[float, str]:
    """
    Return the critical value at alpha. If alpha not in table:
      - 'interpolate': linear interpolation between nearest levels
      - 'nearest': pick nearest level
    Returns (critical_value, note)
    """
    if alpha in levels:
        return float(crits[np.where(levels == alpha)[0][0]]), "exact"

    if policy == "nearest":
        idx = int(np.argmin(np.abs(levels - alpha)))
        return float(crits[idx]), f"nearest({levels[idx]:.3g})"

    if alpha < levels.min():
        idx = np.argmin(levels)
        return float(crits[idx]), f"clamped_low({levels[idx]:.3g})"
    if alpha > levels.max():
        idx = np.argmax(levels)
        return float(crits[idx]), f"clamped_high({levels[idx]:.3g})"

    idx_hi = int(np.searchsorted(levels, alpha, side="left"))
    idx_lo = idx_hi - 1
    a0, a1 = levels[idx_lo], levels[idx_hi]
    c0, c1 = crits[idx_lo], crits[idx_hi]
    t = (alpha - a0) / (a1 - a0)
    c = c0 + t * (c1 - c0)
    return float(c), f"interp({a0:.3g}..{a1:.3g})"


def _compute_ad_statistic(sorted_z: np.ndarray, apply_correction: bool = True) -> float:
    """
    Compute Anderson–Darling A^2 statistic for normality on sorted z-scores.
    If apply_correction=True, apply the common adjustment for estimated mu,sigma.
    """
    n = sorted_z.size
    if n == 0:
        raise ValueError("No data.")
    Fi = norm.cdf(sorted_z)
    eps = np.finfo(float).tiny
    Fi = np.clip(Fi, eps, 1 - eps)
    i = np.arange(1, n + 1, dtype=float)
    A2 = -n - (1.0 / n) * np.sum((2 * i - 1) * (np.log(Fi) + np.log(1.0 - Fi[::-1])))
    if apply_correction and n > 0:
        A2 *= 1.0 + 4.0 / n - 25.0 / (n * n)
    return float(A2)


def run_ad_test(
    data: np.ndarray | pd.Series,
    *,
    alpha: float = 0.05,
    mu_sigma_mode: str = "estimate",
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    standardize: bool = False,
    sample_cap: Optional[int] = None,
    seed: int = 42,
    significance_mode: str = "critical",
    critical_policy: str = "interpolate",
    small_sample_correction: str = "auto",
) -> Dict[str, object]:
    """
    Run Anderson–Darling Normality test with maximum customization.
    Returns a dict with 'summary', 'aux', and 'diagnostics'.
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
            "Input series is constant or near-constant; AD test undefined."
        )

    sampling_applied = False
    if (
        sample_cap is not None
        and isinstance(sample_cap, int)
        and sample_cap > 0
        and x.size > sample_cap
    ):
        x = uniform_sample(x, sample_cap, seed=seed)
        sampling_applied = True

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

    standardized = False
    if standardize:
        x = zscore_inplace(x, mu_used, sigma_used)
        mu_used, sigma_used = 0.0, 1.0
        standardized = True

    z = stable_sort(x)
    n_used = int(z.size)

    apply_corr = (small_sample_correction == "auto") and (mu_sigma_mode == "estimate")
    A2 = _compute_ad_statistic(z, apply_correction=apply_corr)

    backend = None
    p_value: Optional[float] = None
    critical_value: Optional[float] = None
    aux: Optional[Dict[str, object]] = None

    if significance_mode == "pvalue":
        if _HAS_STATSMODELS:
            try:
                stat, p = _sm_normal_ad(z)
                p_value = float(p)
                backend = "statsmodels.normal_ad"
            except Exception:
                backend = "statsmodels_failed_fallback_to_critical"
        else:
            backend = "statsmodels_unavailable_fallback_to_critical"

    if p_value is None:
        levels, crits = _ad_critical_table_for_normal_estimated()
        critical_value, note = _interpolate_critical_value(
            levels, crits, alpha, policy=critical_policy
        )
        decision = (
            "Reject normality" if A2 >= critical_value else "Do not reject normality"
        )
        aux = {
            "critical_value_at_alpha": critical_value,
            "table_levels": levels.tolist(),
            "table_values": crits.tolist(),
            "alpha_mapping": note,
        }
        backend = backend or "scipy:critical-values"
    else:
        decision = "Reject normality" if p_value <= alpha else "Do not reject normality"
        aux = None
        backend = backend or "statsmodels.normal_ad"

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
    if not apply_corr and mu_sigma_mode == "estimate":
        notes.append("Small-sample correction turned off by user.")

    diagnostics = {
        "n_raw": n_raw,
        "n_used": n_used,
        "n_dropped": n_dropped,
        "sampling_applied": sampling_applied,
        "mu_sigma_mode": mu_sigma_mode,
        "mu_used": mu_used,
        "sigma_used": sigma_used,
        "standardized": standardized,
        "significance_mode": significance_mode,
        "critical_policy": critical_policy if p_value is None else None,
        "backend": backend,
        "notes": notes,
    }

    summary = {
        "test": "Anderson–Darling (Normal)",
        "n": n_used,
        "statistic_A2": A2,
        "p_value": p_value,
        "alpha": alpha,
        "decision": decision,
        "critical_value": critical_value,
    }

    return {
        "summary": summary,
        "aux": aux,
        "diagnostics": diagnostics,
    }
