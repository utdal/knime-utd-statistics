# Anderson-Darling & Cramer-von Mises Tests V1.1

A streamlined KNIME Python extension node for performing normality tests with simplified configuration and direct statistical library integration.

## Overview

This node provides normality testing using two established statistical tests: Anderson-Darling and Cramer-von Mises. V1.1 represents a complete simplification from V1.0, removing all customizations in favor of direct library implementations for maximum reliability and minimal complexity.

## Major Changes from V1.0 → V1.1

### **🎯 Simplified Design Philosophy**
- **Removed**: All parameter customization (12+ parameters → 2 parameters)
- **Removed**: Custom statistical implementations (935 lines → 225 lines, 76% reduction)
- **Added**: Direct statsmodels/scipy integration
- **Added**: Strict data validation (rejects data with ANY nulls)

### **⚡ Core Improvements**
- **Performance**: Direct library calls, no custom computation overhead
- **Reliability**: Uses authoritative statistical implementations
- **Maintainability**: Minimal codebase, easier to debug and extend
- **User Experience**: Simple interface, clear error messages

## Features

### 🎯 **Dual Test Support**
- **Anderson-Darling Test**: Via `statsmodels.stats.diagnostic.normal_ad()`
- **Cramer-von Mises Test**: Via `scipy.stats.cramervonmises(data, norm.cdf)`
- **Consistent Interface**: Both tests return identical output format

### 🛡️ **Strict Data Validation**
- **Null Rejection**: Automatically rejects data containing ANY null/missing values
- **Type Validation**: Ensures numeric data types (int/float)
- **Constant Data Check**: Detects and rejects constant/near-constant data
- **Clear Error Messages**: Immediate feedback for validation failures

### 🔧 **Fixed Configuration**
- **Significance Level**: Fixed at α = 0.05 (industry standard)
- **Parameter Estimation**: Always estimates μ/σ from data (no manual override)
- **No Preprocessing**: No data transformations, standardization, or sampling

## Node Configuration

### **Input Requirements**
- **Input Table**: Any KNIME table containing numeric columns
- **Data Quality**: No null values, no constant data, numeric columns only

### **Configuration Parameters**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| **Test Type** | Enum | "Anderson-Darling" or "Cramer-von Mises" | Anderson-Darling |
| **Data column** | Column Selector | Numeric column to test for normality | Required |

**That's it!** Only 2 parameters instead of 12+ in V1.0.

## Output Table

### **Single Results Table**
Clean, focused output with essential information:

| Column | Type | Description |
|--------|------|-------------|
| **Test** | String | "Anderson-Darling" or "Cramer-von Mises" |
| **Column Tested** | String | Name of the analyzed column |
| **Sample Size (n)** | Integer | Number of data points used |
| **Test Statistic** | Double | A² (Anderson-Darling) or W² (Cramer-von Mises) |
| **P-Value** | Double | Exact p-value from statistical library |
| **Statistical Decision** | String | "Reject normality" or "Do not reject normality" |

## Statistical Interpretation

### **Decision Logic**
- **p ≤ 0.05**: Reject normality (strong evidence against normal distribution)
- **p > 0.05**: Do not reject normality (insufficient evidence against normal distribution)

### **P-Value Interpretation Guide**
| P-Value Range | Interpretation | Action |
|---------------|----------------|--------|
| **p > 0.10** | Data looks reasonably normal | Safe to use normal-based methods |
| **0.05 < p ≤ 0.10** | Borderline case | Depends on analysis requirements |
| **0.01 < p ≤ 0.05** | Probably not normal | Investigate further |
| **p ≤ 0.01** | Definitely not normal | Use non-parametric methods |
| **p ≈ 0** | Extremely non-normal | Very strong evidence against normality |

### **Test Differences**
- **Anderson-Darling**: More sensitive to tail deviations (outliers, heavy/light tails)
- **Cramer-von Mises**: More sensitive to center deviations (bimodal, uniform distributions)

## Data Validation Rules

### **Automatic Rejections**
The node will **automatically reject** data that:

1. **Contains ANY null values**: "Column contains N null/missing values. Normality tests require complete data."
2. **Non-numeric types**: "Column must be numeric (int/float). Found: object"
3. **Constant data**: "Column contains only constant values. Normality tests cannot be performed."
4. **Missing column**: "Column not found in input data."

### **No Preprocessing**
Unlike V1.0, V1.1 does **NO** data preprocessing:
- ❌ No null value removal
- ❌ No standardization/z-scoring  
- ❌ No sampling for performance
- ❌ No parameter estimation options
- ❌ No small-sample corrections

**Clean data in → Results out. That's it.**

## Use Cases

### **Quality Assurance**
- Pre-analysis validation before parametric tests (t-tests, ANOVA, regression)
- Manufacturing quality control (verify measurements follow normal distribution)
- Clinical trial data validation

### **Data Science Workflows**
- Feature distribution analysis in machine learning pipelines
- Model assumption verification
- Data preprocessing validation

### **Academic Research**
- Confirmatory data analysis
- Distribution assumption testing
- Comparative studies between Anderson-Darling and Cramer-von Mises

## Technical Implementation

### **Architecture Simplification**
```
V1.0: KNIME → Parameter Processing → Custom Statistics → Complex Output
      (935 lines, 12+ parameters, dual output tables)

V1.1: KNIME → Basic Validation → Library Call → Simple Output  
      (225 lines, 2 parameters, single output table)
```

### **Direct Library Integration**
```python
# Anderson-Darling Test
from statsmodels.stats.diagnostic import normal_ad
statistic, p_value = normal_ad(data)

# Cramer-von Mises Test  
from scipy.stats import cramervonmises, norm
result = cramervonmises(data, norm.cdf)
statistic, p_value = result.statistic, result.pvalue
```

### **Dependencies**
- **Required**: `numpy`, `pandas`, `scipy`, `statsmodels`
- **KNIME**: `knime.extension` API
- **No Optional Dependencies**: All libraries are required

## Error Handling

### **Validation Errors**
- **Clear Messages**: Immediate feedback about why data was rejected
- **No Fallbacks**: Strict validation prevents unreliable results
- **User Guidance**: Error messages explain how to fix data issues

### **No Silent Failures**
Unlike V1.0's graceful fallbacks, V1.1 **explicitly fails** on:
- Missing statistical libraries
- Invalid data types
- Any data quality issues

**Philosophy**: "Fail fast and clearly rather than produce questionable results."

## Performance Characteristics

### **Optimizations Removed**
- ❌ No sampling caps for large datasets
- ❌ No performance monitoring
- ❌ No memory optimization

### **Performance Reality**
- **Small datasets (< 1,000)**: Instant results
- **Medium datasets (1,000 - 10,000)**: Sub-second execution
- **Large datasets (> 10,000)**: May take several seconds

**Philosophy**: "Correctness over performance. Let the libraries handle optimization."

## Migration from V1.0

### **Breaking Changes**
1. **Parameter Reduction**: 12+ parameters → 2 parameters
2. **Output Change**: Dual tables → Single table
3. **Validation Strictness**: Permissive → Strict data requirements
4. **No Customization**: Fixed α=0.05, no parameter overrides

### **Migration Steps**
1. **Update Workflows**: Remove all V1.0 parameter configurations
2. **Clean Data**: Ensure no nulls, verify numeric types
3. **Expect Rejections**: V1.1 may reject data that V1.0 accepted
4. **Update Downstream**: Single output table instead of two

### **Benefits of Migration**
- **Reliability**: Direct library implementations
- **Simplicity**: Minimal configuration
- **Speed**: Faster execution, less overhead
- **Maintainability**: Much easier to debug and modify

## Compatibility

### **KNIME Versions**
- **Minimum**: KNIME 5.4+
- **Tested**: KNIME 5.4, 5.5

### **Python Environment**
- **Minimum**: Python 3.11+
- **Required Packages**: `numpy`, `pandas`, `scipy`, `statsmodels`

### **Operating Systems**
- **Windows**: Tested on Windows 10/11
- **macOS**: Tested on macOS 13+
- **Linux**: Compatible with major distributions

## Version History

### **V1.1.0** (Current)
- **Complete redesign**: Simplified from 935 → 225 lines of code
- **Direct library integration**: statsmodels + scipy implementations
- **Strict validation**: Reject data with any quality issues
- **Dual test support**: Anderson-Darling + Cramer-von Mises
- **Single output table**: Focused, essential results only
- **Fixed configuration**: α=0.05, no parameter customization

### **V1.0.0** (Legacy)
- Complex parameter system with 12+ configuration options
- Custom statistical implementations
- Dual output tables (Results + Diagnostics)
- Graceful fallbacks and data preprocessing
- **Deprecated**: Use V1.1 for new workflows

## References

### **Statistical Libraries**
- **statsmodels**: Seabold, S. & Perktold, J. (2010). "statsmodels: Econometric and statistical modeling with python"
- **scipy**: Virtanen, P. et al. (2020). "SciPy 1.0: fundamental algorithms for scientific computing in Python"

### **Statistical Foundation**
- Anderson, T.W. and Darling, D.A. (1952). "Asymptotic theory of certain 'goodness of fit' criteria"
- Cramér, H. (1928). "On the composition of elementary errors"

### **Implementation Standards**
- KNIME Python Extension API documentation
- Scientific Python coding standards

---

**Author**: UTD Statistics Team  
**License**: See LICENSE.TXT  
**Version**: 1.1.0  
**KNIME Compatibility**: 5.4+  
**Python Requirement**: 3.11+