# ![Image](https://www.knime.com/sites/default/files/knime_logo_github_40x40_4layers.png) KNIME® — UTD Statistical Analysis Extension

[![Code Quality Check](https://github.com/knime/knime-python-extension-template/actions/workflows/code-quality-check.yml/badge.svg)](https://github.com/knime/knime-python-extension-template/actions/workflows/code-quality-check.yml)
[![Extension Bundling](https://github.com/knime/knime-python-extension-template/actions/workflows/bundle-extension.yml/badge.svg)](https://github.com/knime/knime-python-extension-template/actions/workflows/bundle-extension.yml)

Developed by **Ahmed Elghazi** and **Rabih Neouchi** at the [University of Texas at Dallas](https://www.utdallas.edu/).

This KNIME Python extension provides a suite of statistical analysis nodes for hypothesis testing, variance analysis, and regression diagnostics — packaged for seamless use inside KNIME Analytics Platform.

---

## Contents

```
.
├── icons/
│   ├── utd.png                          # Extension category icon
│   ├── curve.jpg                        # Normality Tests icon
│   ├── post_hoc.jpg                     # Post-Hoc Analysis icon
│   ├── heteroskedasticity.png           # Heteroskedasticity Tests icon
│   ├── factorial.jpg                    # Factorial ANOVA icon
│   ├── manova3.png                      # One-Way MANOVA icon
│   └── rm_anova.png                     # Repeated Measures ANOVA icon
├── src/
│   ├── __init__.py
│   ├── normality_node.py
│   ├── normality_tests/
│   ├── post_hoc_node.py
│   ├── post_hoc/
│   ├── heteroskedasticity_node.py
│   ├── heteroskedasticity/
│   ├── factorial_anova_node.py
│   ├── factorial_anova/
│   ├── manova_node.py
│   ├── manova/
│   ├── repeated_measures_anova_node.py
│   └── repeated_measures_anova/
├── test_data/                           # CSV fixtures for unit tests
├── workflows/                           # Demo KNIME workflow
├── knime.yml                            # Extension metadata
├── pixi.toml                            # Dependency management
├── pixi.lock                            # Locked dependency versions
├── ruff.toml                            # Code formatting config
├── pytest.ini                           # Pytest configuration
└── LICENSE.TXT
```

---

## Nodes

### Statistical Normality Tests
Tests whether one or more numeric columns follow a normal distribution.

- **Methods**: Anderson-Darling, Cramer-von Mises
- **Input**: Table with numeric column(s)
- **Output**: Per-column results — Test Statistic, P-Value, Statistical Decision

### Post-Hoc Analysis
Runs a one-way ANOVA and, if significant, identifies which group pairs differ.

- **Methods**: Tukey HSD, Holm-Bonferroni
- **Input**: Numeric dependent variable + categorical grouping variable
- **Outputs**: ANOVA Summary · Pairwise Comparisons (Mean Difference, Corrected P-Value)

### Heteroskedasticity Tests
Checks whether the residual variance of an OLS regression model is constant.

- **Methods**: Breusch-Pagan, White, Goldfeld-Quandt
- **Input**: Numeric target + predictor variables
- **Outputs**: Test Result · Model Summary · Data with Predictions and Residuals

### Factorial ANOVA
Tests whether categorical factors — alone or in combination — significantly affect a continuous outcome.

- **Features**: Up to N-way interactions, Type I/II/III sums of squares, partial eta squared
- **Input**: Numeric response variable + one or more categorical factor variables
- **Outputs**: ANOVA Results (Basic or Advanced) · Model Coefficients with confidence intervals

### One-Way MANOVA
Tests whether group means differ across multiple dependent variables simultaneously.

- **Test statistic**: Pillai's Trace (robust to assumption violations)
- **Input**: Two or more numeric dependent variables + one categorical grouping variable
- **Outputs**: Multivariate Results · Reliability Report (Box's M test)

### Repeated Measures ANOVA
Tests whether the same participants respond differently across conditions or time points.

- **Correction**: Greenhouse-Geisser sphericity correction applied automatically
- **Input format**: Long format only — one row per measurement, with columns for the measured value, condition/time point, and participant ID
- **Output**: Basic summary or full Advanced breakdown (sphericity diagnostics included)

---

## Instructions

### Prerequisites

- [KNIME Analytics Platform](https://www.knime.com/downloads/overview)
- [git](https://git-scm.com/downloads)
- [pixi](https://pixi.sh/latest/)

### Setup

1. **Clone** this repository:
    ```bash
    git clone <repository-url>
    cd knime-utd-statistics
    ```

2. **Review** `knime.yml` to confirm the extension metadata (name, group ID, author, version).

3. **Inspect** the `src/` directory to explore or modify node implementations. Each node is implemented as a Python file (e.g. `factorial_anova_node.py`) backed by a dedicated submodule (e.g. `factorial_anova/`).

4. **Install** the Python environment:
    ```bash
    pixi install
    ```
    This installs all dependencies as defined in `pixi.toml`. The resulting environment is locked in `pixi.lock` — commit both files whenever you add or update packages.

5. _(Optional)_ **Add packages** to the environment:
    ```bash
    pixi add <package_name>
    ```

6. **Register the extension in debug mode** with your local KNIME Analytics Platform:
    ```bash
    pixi run register-debug-in-knime
    ```
    This command auto-detects your KNIME installation and appends the `-Dknime.python.extension.debug_knime_yaml_list` argument to the `knime.ini` file automatically — no manual file editing required. You can then test the nodes in KNIME (see `workflows/` for a demo).

7. **Bundle** your extension:
    ```bash
    pixi run build
    ```
    To place the update site at a custom path (default is `./local-update-site`):
    ```bash
    pixi run build dest=<path_to_your_update_site>
    ```

8. **Install** the bundled extension in KNIME via:
    ```
    File > Install KNIME Extensions... > Available Software Sites > Add...
    ```
    Enter the path to your local update site. After that, install and restart KNIME.

9. To **publish** on KNIME Hub, follow the [KNIME Hub documentation](https://docs.knime.com/latest/development_contribute_extension/index.html#_publish_your_extension).

---

## Testing

Unit tests are located in the repository root (e.g. `test_factorial_anova.py`, `test_repeated_measures_anova.py`). Test data fixtures live in `test_data/`.

Run all tests with:
```bash
pixi run test
```

For a specific test file:
```bash
pixi run python -m pytest test_factorial_anova.py -v
```

---

## Join the Community

- [KNIME Forum](https://forum.knime.com)
- [KNIME Python Extension Documentation](https://docs.knime.com/latest/pure_python_node_extensions_guide/index.html)
