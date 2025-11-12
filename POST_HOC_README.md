# Post-Hoc Multiple Comparisons — Documentation

This document explains the post-hoc node and supporting Python modules in this
repository. It is focused on the `post_hoc` test implementation and the files
that implement ANOVA, Tukey HSD, and Holm-Bonferroni correction.

## Overview

The node performs a one-way ANOVA and, when appropriate, runs post-hoc pairwise
comparisons. It produces **two simplified outputs** for consumption in KNIME:

- **ANOVA Summary** — single-row table with overall test conclusion
- **Pairwise Details** — pairwise comparison results (conditional on ANOVA significance)

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
    - Declares and returns **two KNIME output Schemas**:
      - **Output Port 1 (ANOVA Summary)**: 5 columns
        - Tested Variable (String)
        - Grouping Variable (String)
        - Significance Level (Double)
        - ANOVA p-Value (Double)
        - Overall Conclusion (String)
      - **Output Port 2 (Pairwise Details)**: 5 columns
        - Comparison (String)
        - Post-Hoc Method (String)
        - Mean Difference (Double)
        - Corrected p-Value (Double)
        - Difference is Significant? (String)
  - `execute(exec_ctx, input_table)`
    - Converts KNIME table to pandas, runs ANOVA via `run_one_way_anova()`.
    - Creates **Table 1 (ANOVA Summary)**: single-row DataFrame with overall
      ANOVA results and conclusion ("Significant Difference Found" or 
      "No Difference Found").
    - Creates **Table 2 (Pairwise Details)**: conditional on ANOVA significance:
      - If ANOVA **IS significant** (p ≤ α): runs post-hoc tests and populates
        with actual pairwise comparison data.
      - If ANOVA **IS NOT significant** (p > α): creates single fallback row
        with message "ANOVA not significant (p = [value]). Comparisons were skipped."
    - Converts both DataFrames to KNIME tables and returns them.
  - **REMOVED**: `split_posthoc_results()` function (no longer needed with 
    simplified 2-table output).

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
4. **Table 1 creation**: Single-row ANOVA Summary with conclusion based on 
   whether p-value ≤ α.
5. **Table 2 creation** (conditional):
   - **If ANOVA is significant AND there are >= 3 groups**: Run configured post-hoc test:
     - Tukey: `run_tukey_test()` (statsmodels Tukey) -> `format_tukey_results_for_knime()`
     - Holm: `run_bonferroni_test()` (pairwise t-tests + multipletests) -> `format_bonferroni_results_for_knime()`
     - Extract pairwise comparisons and format into Table 2 with "Yes"/"No" significance flags.
   - **If ANOVA is NOT significant**: Create single fallback row with message 
     stating ANOVA was not significant and comparisons were skipped.
6. Both DataFrames are converted to KNIME tables via `knext.Table.from_pandas()`
   and returned in order (ANOVA Summary, Pairwise Details).

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

## Output Schema Design

The simplified two-table design follows specifications to provide:

1. **ANOVA Summary** (single row):
   - Quick overview of overall test results
   - Clear "Significant Difference Found" or "No Difference Found" conclusion
   - Eliminates need to parse detailed ANOVA source tables

2. **Pairwise Details** (conditional multi-row):
   - Only populated when ANOVA is significant
   - Provides actionable pairwise comparison information
   - Uses "Yes"/"No" flags for easy interpretation
   - When ANOVA is not significant, displays explanatory message instead of 
     empty table or potentially misleading comparisons

## Recommended quick test

Run an example locally to verify the simplified output structure:

```python
import numpy as np
import pandas as pd
from src.post_hoc.anova import run_one_way_anova
from src.post_hoc.tukey_core import run_tukey_test, format_tukey_results_for_knime

np.random.seed(0)
A = np.random.normal(0,1,30)
B = np.random.normal(0.5,1,30)
C = np.random.normal(1.0,1,30)
values = np.concatenate([A,B,C])
groups = np.array(['A']*30 + ['B']*30 + ['C']*30)

# Table 1: ANOVA Summary
anova_results = run_one_way_anova(values, groups, alpha=0.05)
anova_p_value = anova_results["summary"]["p_value"]
conclusion = "Significant Difference Found" if anova_p_value <= 0.05 else "No Difference Found"

table1 = pd.DataFrame({
    'Tested Variable': ['test_column'],
    'Grouping Variable': ['group_column'],
    'Significance Level': [0.05],
    'ANOVA p-Value': [anova_p_value],
    'Overall Conclusion': [conclusion]
})
print("Table 1: ANOVA Summary")
print(table1)

# Table 2: Pairwise Details (conditional)
if anova_p_value <= 0.05:
    out = run_tukey_test(values, groups, alpha=0.05)
    pairwise_df, _ = format_tukey_results_for_knime(out)
    
    table2 = pd.DataFrame({
        'Comparison': pairwise_df['Comparison'],
        'Post-Hoc Method': ['Tukey HSD'] * len(pairwise_df),
        'Mean Difference': pairwise_df['Mean Difference'],
        'Corrected p-Value': pairwise_df.get('Corrected P-Value', pairwise_df.get('P-Value')),
        'Difference is Significant?': ['Yes' if p <= 0.05 else 'No' 
                                        for p in pairwise_df.get('Corrected P-Value', pairwise_df.get('P-Value'))]
    })
else:
    table2 = pd.DataFrame({
        'Comparison': [f"ANOVA not significant (p = {anova_p_value:.3f}). Comparisons were skipped."],
        'Post-Hoc Method': ['N/A'],
        'Mean Difference': [np.nan],
        'Corrected p-Value': [np.nan],
        'Difference is Significant?': ['N/A']
    })

print("\nTable 2: Pairwise Details")
print(table2)
```

## Next steps (optional)
- Add unit tests that verify the two-table output structure and schema compliance.
- Add logging at key points to track ANOVA significance and post-hoc execution paths.
- Consider adding confidence intervals to Pairwise Details output if needed for 
  downstream analysis.

## Version History
- **v2.0**: Simplified to two-table output (ANOVA Summary + Pairwise Details) per 
  specification requirements. Removed unified results DataFrame and split function.
  Added conditional logic for non-significant ANOVA fallback message.
- **v1.0**: Original three-table output (ANOVA Results, Pairwise Comparisons, 
  Group Summary) with unified results DataFrame approach.

---

Documentation updated to reflect simplified two-table output design.
