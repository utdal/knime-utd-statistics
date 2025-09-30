# Anderson-Darling Test V1

A comprehensive KNIME Python extension node for performing Anderson-Darling normality tests with advanced configuration options and robust statistical computation.

## Overview

The Anderson-Darling test is a statistical test that assesses whether a dataset follows a normal distribution. This implementation provides a powerful, user-friendly KNIME node that bridges advanced statistical computation with visual workflow design.

## Features

### 🎯 **Core Functionality**
- **Anderson-Darling Normality Test**: Statistical assessment of normal distribution fit
- **Dual Output Tables**: Results summary + comprehensive diagnostics
- **Flexible Parameter Estimation**: Choose between data-driven or user-specified μ/σ
- **Multiple Significance Modes**: Critical values (fast) or p-values (precise)

### 🔧 **Advanced Configuration**
- **Data Preprocessing**: Automatic handling of missing/infinite values
- **Standardization**: Optional z-score transformation
- **Performance Optimization**: Sampling cap for large datasets
- **Small-Sample Correction**: Automatic adjustment for estimated parameters
- **Critical Value Interpolation**: Handle non-standard significance levels

### 🛡️ **Robustness**
- **Graceful Fallbacks**: Automatic fallback from p-values to critical values
- **Comprehensive Validation**: Data quality checks and parameter validation
- **Error Recovery**: Handles edge cases (constant data, missing values)
- **Deterministic Sampling**: Reproducible results with seed control

## Node Configuration

### **Input Requirements**
- **Input Table**: Any KNIME table containing numeric columns
- **Column Selection**: Automatically filtered to show only numeric columns (int32, int64, double)

### **Configuration Parameters**

#### **Data Settings**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| Data column | Column Selector | Numeric column to test for normality | Required |
| Alpha (significance level) | Double | Decision threshold (0.000001 - 0.5) | 0.05 |
| Sampling cap (rows) | Integer | Optional performance limit for large datasets | 0 (disabled) |
| Random seed | Integer | Deterministic sampling reproducibility | 42 |

#### **Parameter Estimation**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| μ/σ source | Enum | "Estimate from data" or "User specified" | Estimate from data |
| μ (mean) | Double | User-provided mean (if user-specified mode) | 0.0 |
| σ (std) | Double | User-provided standard deviation (if user-specified mode) | 1.0 |
| Standardize | Boolean | Z-score data before testing | False |

#### **Test Configuration**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| Significance mode | Enum | "Critical values (SciPy)" or "p-value (statsmodels)" | Critical values |
| Critical-value policy | Enum | "Interpolate" or "Nearest level" for non-tabulated α | Interpolate |
| Small-sample correction | Enum | "Auto" or "Off" for parameter estimation adjustment | Auto |

## Output Tables

### **Port 0: Results Table**
Single-row summary of the Anderson-Darling test:

| Column | Type | Description |
|--------|------|-------------|
| test | String | Test identifier: "Anderson–Darling (Normal)" |
| column | String | Name of the tested column |
| n | Integer | Sample size used in the test |
| statistic_A2 | Double | Anderson-Darling A² test statistic |
| p_value | Double | P-value (if computed) or NaN |
| alpha | Double | Significance level used |
| decision | String | "Reject normality" or "Do not reject normality" |
| critical_value | Double | Critical value (if used) or NaN |

### **Port 1: Diagnostics Table**
Key-value pairs providing complete test transparency:

| Key Category | Example Keys | Description |
|--------------|--------------|-------------|
| **Data Quality** | n_raw, n_used, n_dropped | Data preprocessing summary |
| **Parameters** | mu_used, sigma_used, mu_sigma_mode | Parameter estimation details |
| **Configuration** | significance_mode, backend, critical_policy | Test methodology used |
| **Performance** | sampling_applied | Performance optimization applied |
| **Metadata** | table_levels, alpha_mapping | Statistical reference information |
| **Notes** | note | Warnings and contextual information |

## Statistical Interpretation

### **Decision Logic**
- **Critical Value Mode**: A² ≥ critical_value → Reject normality
- **P-value Mode**: p_value ≤ α → Reject normality

### **Test Statistic (A²)**
- **Low values (< 0.5)**: Good evidence of normality
- **Moderate values (0.5-1.0)**: Weak evidence against normality
- **High values (> 1.0)**: Strong evidence against normality

### **Sample Size Considerations**
- **Small samples (n < 8)**: Low statistical power, higher sensitivity to outliers
- **Large samples (n > 5000)**: High power, may detect trivial deviations from normality

## Use Cases

### **Quality Control**
- Validate normality assumptions before parametric statistical tests
- Process control in manufacturing (check if measurements follow normal distribution)
- Regulatory compliance for normally-distributed requirements

### **Data Science**
- Preprocessing step for machine learning pipelines
- Feature engineering validation
- Model assumption verification

### **Research Applications**
- Confirmatory analysis with known population parameters
- Exploratory data analysis for distribution assessment
- Comparative studies between different normality tests

## Technical Implementation

### **Architecture**
```
KNIME Node Layer → Parameter Translation → Statistical Core → Output Formatting
     ↓                     ↓                      ↓               ↓
UI Parameters → Function Arguments → run_ad_test() → Results + Diagnostics
```

### **Key Algorithms**
- **Anderson-Darling Formula**: Implemented with small-sample correction
- **Critical Value Interpolation**: Linear interpolation for non-tabulated α levels
- **Robust Data Processing**: NaN/Inf handling with stable sorting
- **Parameter Estimation**: Unbiased sample statistics (ddof=1)

### **Dependencies**
- **Core**: `numpy`, `pandas`, `scipy.stats`
- **Optional**: `statsmodels` (for p-value computation)
- **KNIME**: `knime.extension` API

## Error Handling

### **Automatic Recovery**
- **Missing statsmodels**: Falls back to critical values with user warning
- **Invalid data**: Removes NaN/Inf values automatically
- **Small samples**: Provides warnings about reduced statistical power

### **User Validation**
- **Column selection**: Ensures numeric column is selected
- **Parameter consistency**: Validates user-specified μ/σ values
- **Data quality**: Checks for constant/near-constant data

## Performance Optimization

### **Large Dataset Handling**
- **Sampling Cap**: Uniform random sampling for datasets > specified threshold
- **Deterministic Results**: Seed-controlled sampling for reproducibility
- **Memory Efficiency**: Streaming data processing where possible

### **Computational Efficiency**
- **Stable Sorting**: O(n log n) mergesort for numerical stability
- **Vectorized Operations**: NumPy-optimized statistical computations
- **Minimal Memory Footprint**: In-place operations where safe

## Version History

### **V1.0 Features**
- Complete Anderson-Darling implementation
- Dual significance modes (critical values + p-values)
- Comprehensive parameter control
- Robust error handling and edge case management
- Performance optimization for large datasets
- Extensive diagnostics and transparency

## References

### **Statistical Foundation**
- Anderson, T.W. and Darling, D.A. (1952). "Asymptotic theory of certain 'goodness of fit' criteria based on stochastic processes"
- Stephens, M.A. (1974). "EDF Statistics for Goodness of Fit and Some Comparisons"

### **Implementation References**
- SciPy statistical functions documentation
- KNIME Python Extension API guidelines
- NumPy numerical computing best practices

---

**Author**: UTD Statistics Team  
**License**: See LICENSE.TXT  
**KNIME Version**: Compatible with KNIME 5.4+  
**Python Version**: Requires Python 3.11+