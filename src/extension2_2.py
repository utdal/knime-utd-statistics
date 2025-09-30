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

# KNIME Extension API
import knime.extension as knext

# ────────────────────────────────────────────────────────────────
# 1) KNIME node wiring (parameters, configure, execute)
# ────────────────────────────────────────────────────────────────

# Category: adjust to your extension structure
ad_category = knext.category(
    path="/community",
    level_id="normality_tests",
    name="Normality Tests (Python)",
    description="Normality test nodes implemented in Python.",
    icon="./icons/icon.png",
)


def _is_numeric(col: knext.Column) -> bool:
    """Helper function to filter for numeric columns."""
    return col.ktype in (knext.double(), knext.int32(), knext.int64())


@knext.parameter_group("Anderson–Darling Settings")
class _ADParams:
    """Class to define all parameters for the Anderson–Darling node."""
    
    # Data
    input_column = knext.ColumnParameter(
        label="Data column",
        description="Numeric column to test for normality.",
        column_filter=_is_numeric,
    )

    # Alpha
    alpha = knext.DoubleParameter(
        label="Alpha (significance level)",
        description="Decision threshold (e.g., 0.05).",
        default_value=0.05,
        min_value=1e-6,
        max_value=0.5,
    )

    # Mu/Sigma handling
    class _MuSigma(knext.EnumParameterOptions):
        ESTIMATE = ("Estimate from data", "Estimate μ, σ from the selected column.")
        USER     = ("User specified",     "Use user-provided μ and σ.")

    mu_sigma_mode = knext.EnumParameter(
        label="μ/σ source",
        description="Choose whether to estimate μ,σ from the data or provide them.",
        enum=_MuSigma,
        default_value=_MuSigma.ESTIMATE.name,
    )

    mu = knext.DoubleParameter(
        label="μ (mean) if user-specified",
        description="Used only if μ/σ source = User specified.",
        default_value=0.0,
        # display_condition=knext.node_parameter_value("mu_sigma_mode", _MuSigma.USER.name),
    )

    sigma = knext.DoubleParameter(
        label="σ (std) if user-specified",
        description="Must be > 0 when μ/σ source = User specified.",
        default_value=1.0,
        min_value=1e-12,
        # display_condition=knext.node_parameter_value("mu_sigma_mode", _MuSigma.USER.name),
    )

    # Standardization
    standardize = knext.BoolParameter(
        label="Standardize (z-score) data before test",
        description="Subtract μ and divide by σ prior to computation.",
        default_value=False,
    )

    # Significance mapping: critical vs p-value
    class _SigMode(knext.EnumParameterOptions):
        CRITICAL = ("Critical values (SciPy)", "Decision via critical values; no p-value.")
        PVALUE   = ("p-value (statsmodels)",   "Uses statsmodels to compute an AD p-value (if available).")

    significance_mode = knext.EnumParameter(
        label="Significance mode",
        description="Choose how to map the AD statistic to a decision.",
        enum=_SigMode,
        default_value=_SigMode.CRITICAL.name,
    )

    # Critical policy (only relevant in critical mode)
    class _CritPolicy(knext.EnumParameterOptions):
        INTERP  = ("Interpolate", "Linearly interpolate critical value for non-tabulated alpha.")
        NEAREST = ("Nearest level", "Use nearest tabulated alpha level.")

    critical_policy = knext.EnumParameter(
        label="Critical-value policy",
        description="How to get critical value if alpha is not a tabulated level.",
        enum=_CritPolicy,
        default_value=_CritPolicy.INTERP.name,
        # display_condition=knext.node_parameter_value("significance_mode", _SigMode.CRITICAL.name),
    )

    # Small-sample correction
    class _SSCorr(knext.EnumParameterOptions):
        AUTO = ("Auto", "Apply the common AD correction for estimated parameters.")
        OFF  = ("Off",  "Do not apply the small-sample correction.")

    small_sample_correction = knext.EnumParameter(
        label="Small-sample correction",
        description="Adjustment used when parameters are estimated from data.",
        enum=_SSCorr,
        default_value=_SSCorr.AUTO.name,
    )

    # Performance
    sample_cap = knext.IntParameter(
        label="Sampling cap (rows)",
        description="Optional: uniformly sample to this many rows for performance. Leave empty or 0 for no cap.",
        default_value=0,
        min_value=0,
    )

    seed = knext.IntParameter(
        label="Random seed (sampling)",
        description="Used only if a sampling cap is applied.",
        default_value=42,
    )


@knext.node(
    name="Anderson–Darling Normality (Python)",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="./icons/icon.png",
    category=ad_category,
)
@knext.input_table(
    name="Input data",
    description="Table containing the numeric column to test."
)
@knext.output_table(
    name="Results",
    description="One-row table with statistic, p-value (if available), alpha, decision, critical value."
)
@knext.output_table(
    name="Diagnostics",
    description="Key/value diagnostics and notes about preprocessing and backend."
)
class AndersonDarlingNode:
    """Runs the Anderson–Darling normality test on a selected numeric column."""

    params = _ADParams()

    def configure(self, cfg_ctx, input_spec):
        """
        Configures the node's output table schema based on the input.
        This method is called when the user opens the configuration dialog.
        """
        if self.params.significance_mode == _ADParams._SigMode.PVALUE.name and not _HAS_STATSMODELS:
            cfg_ctx.set_warning(
                "Significance mode is set to p-value, but 'statsmodels' is not available. "
                "The node will fall back to critical-value mode at execute time."
            )

        results_cols = [
            knext.Column(knext.string(), "test"),
            knext.Column(knext.string(), "column"),
            knext.Column(knext.int32(),  "n"),
            knext.Column(knext.double(), "statistic_A2"),
            knext.Column(knext.double(), "p_value"),
            knext.Column(knext.double(), "alpha"),
            knext.Column(knext.string(), "decision"),
            knext.Column(knext.double(), "critical_value"),
        ]
        diag_cols = [
            knext.Column(knext.string(), "key"),
            knext.Column(knext.string(), "value"),
        ]

        # Return schemas for both output tables as a list
        results_schema = knext.Schema.from_columns(results_cols)
        diag_schema = knext.Schema.from_columns(diag_cols)
        return [results_schema, diag_schema]

    def execute(self, exec_ctx, input_table):
        """
        Executes the node's core logic.
        This method is called when the node is run.
        """
        df = input_table.to_pandas()
        col_name = self.params.input_column
        
        # Guard against no column selection
        if col_name is None:
            raise ValueError("No column selected. Please configure the node and select a numeric data column.")
        
        # Guard against invalid column selection
        if col_name not in df.columns:
            raise ValueError(f"Selected column '{col_name}' not found in input data.")

        # Resolve UI parameters -> core function arguments
        alpha = self.params.alpha
        mu_sigma_mode = "estimate" if self.params.mu_sigma_mode == _ADParams._MuSigma.ESTIMATE.name else "user"
        mu = self.params.mu if mu_sigma_mode == "user" else None
        sigma = self.params.sigma if mu_sigma_mode == "user" else None
        standardize = bool(self.params.standardize)
        sample_cap = int(self.params.sample_cap) if int(self.params.sample_cap) > 0 else None
        seed = int(self.params.seed)
        significance_mode = "critical" if self.params.significance_mode == _ADParams._SigMode.CRITICAL.name else "pvalue"
        critical_policy = "interpolate" if self.params.critical_policy == _ADParams._CritPolicy.INTERP.name else "nearest"
        small_sample_correction = "auto" if self.params.small_sample_correction == _ADParams._SSCorr.AUTO.name else "off"

        # Call computational core
        res = run_ad_test(
            df[col_name],
            alpha=alpha,
            mu_sigma_mode=mu_sigma_mode,
            mu=mu,
            sigma=sigma,
            standardize=standardize,
            sample_cap=sample_cap,
            seed=seed,
            significance_mode=significance_mode,
            critical_policy=critical_policy,
            small_sample_correction=small_sample_correction,
        )

        # Build Results table (1 row)
        summary = res["summary"]
        results_df = pd.DataFrame([{
            "test": summary["test"],
            "column": col_name,
            "n": np.int32(summary["n"]),
            "statistic_A2": summary["statistic_A2"],
            "p_value": float('nan') if summary["p_value"] is None else summary["p_value"],
            "alpha": summary["alpha"],
            "decision": summary["decision"],
            "critical_value": float('nan') if summary["critical_value"] is None else summary["critical_value"]
        }])

        # Build Diagnostics key/value table
        diag = res["diagnostics"]
        rows = []
        def _put(k, v):
            rows.append({"key": str(k), "value": "" if v is None else str(v)})

        _put("n_raw", diag["n_raw"])
        _put("n_used", diag["n_used"])
        _put("n_dropped", diag["n_dropped"])
        _put("sampling_applied", diag["sampling_applied"])
        _put("mu_sigma_mode", diag["mu_sigma_mode"])
        _put("mu_used", diag["mu_used"])
        _put("sigma_used", diag["sigma_used"])
        _put("standardized", diag["standardized"])
        _put("significance_mode", diag["significance_mode"])
        _put("critical_policy", diag["critical_policy"])
        _put("backend", diag["backend"])
        for note in diag.get("notes", []):
            _put("note", note)
        
        if res.get("aux") is not None:
            aux = res["aux"]
            _put("critical_value_at_alpha", aux.get("critical_value_at_alpha"))
            _put("alpha_mapping", aux.get("alpha_mapping"))
            _put("table_levels", aux.get("table_levels"))
            _put("table_values", aux.get("table_values"))

        diag_df = pd.DataFrame(rows, columns=["key", "value"])

        return (knext.Table.from_pandas(results_df), knext.Table.from_pandas(diag_df))


# ────────────────────────────────────────────────────────────────
# 2) Computational core (pure Python; unit-testable)
# ────────────────────────────────────────────────────────────────

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
    crits  = np.array([0.576, 0.656, 0.787, 0.918, 1.092], dtype=float)
    return levels, crits


def _interpolate_critical_value(levels: np.ndarray, crits: np.ndarray, alpha: float,
                                policy: str = "interpolate") -> Tuple[float, str]:
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
        A2 *= (1.0 + 4.0 / n - 25.0 / (n * n))
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
    small_sample_correction: str = "auto"
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
        raise ValueError("No valid observations after removing missing/non-finite values.")
    
    if np.allclose(np.std(x, ddof=1 if x.size > 1 else 0), 0.0, rtol=0, atol=1e-15):
        raise ValueError("Input series is constant or near-constant; AD test undefined.")

    sampling_applied = False
    if sample_cap is not None and isinstance(sample_cap, int) and sample_cap > 0 and x.size > sample_cap:
        x = uniform_sample(x, sample_cap, seed=seed)
        sampling_applied = True

    if mu_sigma_mode not in ("estimate", "user"):
        raise ValueError("mu_sigma_mode must be 'estimate' or 'user'.")
    if mu_sigma_mode == "estimate":
        mu_used, sigma_used = sample_mean_std(x)
        if not np.isfinite(sigma_used) or sigma_used <= 0:
            raise ValueError("Estimated standard deviation is non-positive; cannot proceed.")
    else:
        if mu is None or sigma is None or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("When mu_sigma_mode='user', provide finite mu and sigma>0.")
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
        critical_value, note = _interpolate_critical_value(levels, crits, alpha, policy=critical_policy)
        decision = "Reject normality" if A2 >= critical_value else "Do not reject normality"
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

# ────────────────────────────────────────────────────────────────
# 3) Main function for testing the computational core
# ────────────────────────────────────────────────────────────────

def _test_case(title, data_path, **kwargs):
    """Helper function to run a test case and print the results."""
    print(f"\n--- {title} ---")
    try:
        data = pd.read_csv(data_path)["x"]
        result = run_ad_test(data, **kwargs)
        summary = result["summary"]
        diagnostics = result["diagnostics"]
        
        print(f"  Result: {summary['decision']} (alpha={summary['alpha']})")
        print(f"  Statistic: A² = {summary['statistic_A2']:.4f}")
        if summary.get("p_value") is not None:
            print(f"  P-value: {summary['p_value']:.4f}")
        if summary.get("critical_value") is not None:
            print(f"  Critical Value: {summary['critical_value']:.4f}")
        
        print("\n  Notes:")
        for note in diagnostics["notes"]:
            print(f"    - {note}")
            
    except Exception as e:
        print(f"  Test failed as expected: {e}")
        
    print("-" * (len(title) + 8))


if __name__ == "__main__":
    print("Running Anderson-Darling test cases using the computational core.\n")
    print("This will test data loading, parameter handling, and edge cases.")
    print("Note: 'statsmodels' required for p-value tests.")

    # Test Case 1: Ideal Normal Data
    _test_case(
        "Normal Distribution",
        "testData/ad_test_data_normal.csv",
        significance_mode="pvalue"
    )

    # Test Case 2: Lognormal Data (should reject)
    _test_case(
        "Lognormal Distribution",
        "testData/ad_test_data_lognorm.csv",
        significance_mode="pvalue"
    )

    # Test Case 3: Student-t(3) Data (should reject)
    _test_case(
        "Student's t(3) Distribution",
        "testData/ad_test_data_student_t3.csv",
        significance_mode="pvalue"
    )
    
    # Test Case 4: Normal Mixture Model (should reject)
    _test_case(
        "Normal Mixture",
        "testData/ad_test_data_mixture.csv",
        significance_mode="pvalue"
    )
    
    # Test Case 5: Small sample size (should run, but with a warning)
    _test_case(
        "Small Sample (n=7)",
        "testData/ad_test_data_smalln.csv"
    )
    
    # Test Case 6: Known mu/sigma
    _test_case(
        "Known mu/sigma (Test against N(5, 2))",
        "testData/ad_test_data_known_mu_sigma.csv",
        mu_sigma_mode="user",
        mu=5.0,
        sigma=2.0
    )

    # Test Case 7: Data with missing values
    _test_case(
        "Data with NaNs and empty cells",
        "testData/ad_test_data_with_nans.csv"
    )
    
    # Test Case 8: Data with non-finite values
    _test_case(
        "Data with +/- inf",
        "testData/ad_test_data_with_infs.csv"
    )

    # Test Case 9: Constant data (should raise ValueError)
    _test_case(
        "Constant Data",
        "testData/ad_test_data_constant.csv"
    )
    
    # Test Case 10: Huge dataset with sampling cap
    _test_case(
        "Huge Dataset (with sampling cap)",
        "testData/ad_test_data_huge.csv",
        sample_cap=1000
    )