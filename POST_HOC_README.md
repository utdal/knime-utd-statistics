# Post-Hoc Multiple Comparisons — Documentation

This document explains the post-hoc node and supporting Python modules in this
repository. It is focused on the `post_hoc` test implementation and the files
that implement ANOVA, Tukey HSD, and Holm-Bonferroni correction.

## Overview

The node performs a one-way ANOVA and, when appropriate, runs post-hoc pairwise
comparisons. It produces three outputs for consumption in KNIME:

- ANOVA Results — overall F-test summary
- Pairwise Comparisons — pairwise results with raw and corrected p-values
- Group Summary — descriptive statistics per group with metadata

## Files (concise, file-by-file)

### src/post_hoc_node.py
- Role: KNIME node orchestration and I/O.
- Key imports: `knime.extension` (KNIME API), `numpy`, `pandas`, and helpers from
  `src.post_hoc` (run_one_way_anova, formatters, post-hoc test runners, params).
- Main pieces:
  - `_validate_and_prepare_data(df, data_col, group_col)`
    - Validates that the selected columns exist and that data satisfy ANOVA
      preconditions; returns numpy arrays (data, groups).
  - `configure(cfg_ctx, input_spec)`
    - Declares and returns the three KNIME output Schemas (ANOVA, Pairwise,
      Group Summary).
  - `execute(exec_ctx, input_table)`
    - Converts KNIME table to pandas, runs ANOVA, conditionally runs post-hoc,
      assembles unified results, coerces dtypes & column ordering, splits into
      three DataFrames, converts to KNIME tables and returns them.
  - `split_posthoc_results(results_df)`
    - Splits the unified DataFrame into three DataFrames and coerces types.

### src/post_hoc/tukey_core.py
- Role: Run Tukey HSD (statsmodels) and compute raw diagnostics (Welch t-tests).
- Key imports: `pairwise_tukeyhsd` from `statsmodels`, `scipy.stats` for
  `ttest_ind`, `itertools.combinations` for deterministic pair ordering.
- Main functions:
  - `run_tukey_test(data, groups, alpha=0.05)`
    - Drops missing, calls `pairwise_tukeyhsd` for corrected p-values and reject
      flags, computes raw Welch t-test p-values and t-statistics for each pair,
      assembles `results` DataFrame and `group_summary` DataFrame, returns dict.
  - `format_tukey_results_for_knime(tukey_output)`
    - Ensures columns exist, coerces numeric types, adds `Test Method` and
      `Significance Level`, returns `(pairwise_df, group_summary_df)`.

### src/post_hoc/bonferroni_core.py
- Role: Run pairwise Welch t-tests and apply Holm sequential correction.
- Key imports: `scipy.stats.ttest_ind`, `statsmodels.stats.multitest.multipletests`.
- Main functions:
  - `run_bonferroni_test(data, groups, alpha=0.05)`
    - Runs all pairwise Welch t-tests, stores raw p-values and statistics,
      applies `multipletests(..., method='holm')` to compute corrected p-values
      and reject flags, computes approximate confidence intervals, builds
      `results` DataFrame and `group_summary` DataFrame, returns dict.
  - `format_bonferroni_results_for_knime(bonferroni_output)`
    - Mirrors raw P-Value to 'P-Value', adds `Test Method` and `Significance Level`,
      coerces types and returns `(pairwise_df, group_summary_df)`.

## Data flow (end-to-end)
1. KNIME input table -> `input_table.to_pandas()` -> pandas DataFrame.
2. Validation: `_validate_and_prepare_data()` extracts numpy arrays (data, groups)
   and checks assumptions required by ANOVA.
3. ANOVA: `run_one_way_anova(data, groups)` computes F-statistic and p-value.
4. If the ANOVA is significant and there are >= 3 groups, the node runs the
   configured post-hoc test:
   - Tukey: `run_tukey_test()` (statsmodels Tukey) -> `format_tukey_results_for_knime()`
   - Holm: `run_bonferroni_test()` (pairwise t-tests + multipletests) -> `format_bonferroni_results_for_knime()`
5. The node creates a unified results list of dicts containing ANOVA rows,
   pairwise rows, and group-summary rows. It coerces exact schema order & types
   (matching `configure()`), then calls `split_posthoc_results()` to produce
   three DataFrames which are converted to KNIME tables and returned.

## Why scipy.stats is used
- `scipy.stats.ttest_ind(..., equal_var=False)` (Welch t-test) is used to
  compute raw pairwise t-statistics and p-values. This provides diagnostics
  and parity with the Holm correction path which explicitly performs pairwise
  t-tests. ANOVA is an F-test and does not produce pairwise t-statistics.

## Why itertools is used
- `itertools.combinations()` produces a deterministic ordering of group pairs
  (e.g., (g1,g2), (g1,g3), (g2,g3)) that matches the ordering of arrays
  returned by `pairwise_tukeyhsd`. It ensures consistent alignment when
  combining Tukey arrays with separately computed raw p-values.

## Where columns can be lost (diagnostics)
- The `split_posthoc_results()` helper selects explicit column lists for each
  output. If a metadata column (e.g., 'Test Method' or 'Significance Level')
  is not included in those lists it will be dropped at split time and later
  reintroduced as empty (NaN or empty string) when the node enforces the
  configured schema. Recent edits ensure that 'Test Method' and 'Significance Level'
  are preserved for Group Summary.

## Recommended quick test

Run an example locally to see pairwise and group summary outputs:

```python
import numpy as np
from src.post_hoc.tukey_core import run_tukey_test, format_tukey_results_for_knime

np.random.seed(0)
A = np.random.normal(0,1,30)
B = np.random.normal(0.5,1,30)
C = np.random.normal(1.0,1,30)
values = np.concatenate([A,B,C])
groups = np.array(['A']*30 + ['B']*30 + ['C']*30)

out = run_tukey_test(values, groups, alpha=0.05)
pairwise, group_summary = format_tukey_results_for_knime(out)
print(pairwise)
print(group_summary)
```

## Next steps (optional)
- Add unit tests that verify the presence of metadata columns in all outputs.
- Add logging at key points (pre- and post- split) to debug missing data issues.

---

Documentation generated per request.
