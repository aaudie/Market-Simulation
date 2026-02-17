# Validation & Testing Framework

**Document Type:** Implementation Guide  
**Thesis Relevance:** High (Results & Discussion)  
**Last Updated:** February 2026

---

## Executive Summary

This document provides a comprehensive validation framework for the market simulation system. Implement these tests to demonstrate rigor and build confidence in results.

---

## 1. Validation Hierarchy

```
Level 1: Unit Tests (Component-level)
├─ Transition matrix properties (rows sum to 1)
├─ Regime inference consistency
└─ Numerical stability

Level 2: Integration Tests (System-level)
├─ End-to-end simulation runs
├─ Monte Carlo convergence
└─ Parameter sensitivity

Level 3: Statistical Validation (Model-level)
├─ In-sample fit
├─ Out-of-sample prediction
├─ Cross-validation
└─ Bootstrap robustness

Level 4: Economic Validation (Interpretability)
├─ Historical event detection
├─ Theoretical consistency
└─ Expert judgment alignment
```

---

## 2. Quick Implementation Guide

### 2.1 Basic Validation (1-2 days)

```python
# Script: validation/basic_tests.py

def validate_transition_matrix(P):
    """Ensure transition matrix is valid."""
    # Check: rows sum to 1
    assert np.allclose(P.sum(axis=1), 1.0), "Rows must sum to 1"
    
    # Check: all probabilities in [0, 1]
    assert np.all(P >= 0) and np.all(P <= 1), "Invalid probabilities"
    
    print("✓ Transition matrix valid")

def validate_regime_inference(model, known_crises):
    """Validate against known historical events."""
    correct = 0
    for event, info in known_crises.items():
        predicted = model.infer_regime(info['data'])
        expected = info['regime']
        if predicted == expected:
            correct += 1
            print(f"✓ {event}: {predicted}")
        else:
            print(f"✗ {event}: predicted {predicted}, expected {expected}")
    
    accuracy = correct / len(known_crises)
    print(f"\nCrisis detection accuracy: {accuracy:.1%}")
    return accuracy

# Known crises
KNOWN_CRISES = {
    '2008 Financial Crisis': {
        'date': '2008-10',
        'regime': 'panic',
        'data': crisis_data_2008
    },
    '2020 COVID-19': {
        'date': '2020-03',
        'regime': 'panic',
        'data': crisis_data_2020
    },
    # ... add more
}

# Run validation
validate_transition_matrix(model.P_traditional)
validate_regime_inference(model, KNOWN_CRISES)
```

---

### 2.2 Cross-Validation (2-3 days)

```python
# Script: validation/cross_validation.py

from sklearn.model_selection import TimeSeriesSplit

def time_series_cross_validate(model_class, data, n_splits=5):
    """
    Expanding window cross-validation for time series.
    
    Split 1: Train[1950-2000] → Test[2000-2005]
    Split 2: Train[1950-2005] → Test[2005-2010]
    etc.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scores = []
    for train_idx, test_idx in tscv.split(data):
        # Split data
        train_data = data[train_idx]
        test_data = data[test_idx]
        
        # Train model
        model = model_class()
        model.fit(train_data)
        
        # Evaluate
        score = model.evaluate(test_data)
        scores.append(score)
        print(f"Fold: Train accuracy={model.evaluate(train_data):.3f}, "
              f"Test accuracy={score:.3f}")
    
    # Summary
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"\nCross-validation results:")
    print(f"  Mean test score: {mean_score:.3f} ± {std_score:.3f}")
    print(f"  Min/Max: [{min(scores):.3f}, {max(scores):.3f}]")
    
    return {'scores': scores, 'mean': mean_score, 'std': std_score}

# Usage
cv_results = time_series_cross_validate(
    HybridMarkovModel,
    training_data,
    n_splits=5
)
```

---

### 2.3 Bootstrap Confidence Intervals (3-4 days)

```python
# Script: validation/bootstrap.py

def bootstrap_confidence_interval(model, data, metric_func, 
                                 n_bootstrap=1000, confidence=0.95):
    """
    Bootstrap CI for any metric.
    
    Args:
        model: Model instance
        data: Training data
        metric_func: Function(model) → scalar metric
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 = 95%)
    
    Returns:
        {'estimate': float, 'ci_lower': float, 'ci_upper': float}
    """
    bootstrap_metrics = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(len(data), size=len(data), replace=True)
        resampled_data = data[indices]
        
        # Refit model
        model_copy = model.__class__()
        model_copy.fit(resampled_data)
        
        # Calculate metric
        metric = metric_func(model_copy)
        bootstrap_metrics.append(metric)
        
        if (i+1) % 100 == 0:
            print(f"  Bootstrap {i+1}/{n_bootstrap}")
    
    # Calculate CI
    alpha = 1 - confidence
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100
    
    result = {
        'estimate': np.mean(bootstrap_metrics),
        'ci_lower': np.percentile(bootstrap_metrics, lower_pct),
        'ci_upper': np.percentile(bootstrap_metrics, upper_pct),
        'distribution': bootstrap_metrics
    }
    
    print(f"\nBootstrap results ({confidence*100:.0f}% CI):")
    print(f"  Estimate: {result['estimate']:.4f}")
    print(f"  CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    
    return result

# Example: CI for calm→neutral transition probability
def calm_to_neutral_prob(model):
    probs = model.predict_next_regime('calm', {'is_tokenized': 0.0})[1]
    return probs[1]  # Index 1 = neutral

ci = bootstrap_confidence_interval(
    model=hybrid_model,
    data=training_data,
    metric_func=calm_to_neutral_prob,
    n_bootstrap=1000
)
```

---

## 3. Statistical Tests

### 3.1 Regime Distribution Comparison

```python
# Test: Do traditional and tokenized markets have different regime distributions?

from scipy.stats import chi2_contingency

def test_regime_difference(regimes_trad, regimes_token):
    """
    Chi-square test for regime distribution equality.
    
    H0: Traditional and tokenized have same regime distribution
    H1: Distributions differ
    """
    # Count regimes
    regime_names = ['calm', 'neutral', 'volatile', 'panic']
    
    counts_trad = [sum(regimes_trad == r) for r in regime_names]
    counts_token = [sum(regimes_token == r) for r in regime_names]
    
    # Contingency table
    observed = np.array([counts_trad, counts_token])
    
    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(observed)
    
    print(f"Chi-square test for regime differences:")
    print(f"  χ² = {chi2:.2f}, df = {dof}, p = {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  → Reject H0: Distributions differ (p < 0.05)")
    else:
        print(f"  → Fail to reject H0: No evidence of difference")
    
    return {'chi2': chi2, 'p_value': p_value, 'dof': dof}

# Run test
test_regime_difference(traditional_regimes, tokenized_regimes)
```

---

### 3.2 Transition Probability Significance

```python
def test_transition_difference(P_trad, P_token, from_regime, to_regime, 
                               n_trad, n_token):
    """
    Test if transition probabilities differ significantly.
    
    Uses normal approximation for proportions.
    """
    i = regime_to_index[from_regime]
    j = regime_to_index[to_regime]
    
    p1 = P_trad[i, j]
    p2 = P_token[i, j]
    
    # Standard errors (from binomial)
    se1 = np.sqrt(p1 * (1 - p1) / n_trad)
    se2 = np.sqrt(p2 * (1 - p2) / n_token)
    
    # Test statistic
    z = (p1 - p2) / np.sqrt(se1**2 + se2**2)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    print(f"Test: {from_regime} → {to_regime}")
    print(f"  Traditional: {p1:.3f} ± {1.96*se1:.3f}")
    print(f"  Tokenized:   {p2:.3f} ± {1.96*se2:.3f}")
    print(f"  Difference:  {p1-p2:.3f}")
    print(f"  z = {z:.2f}, p = {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  → Significant difference (p < 0.05)")
    else:
        print(f"  → No significant difference")
    
    return {'z': z, 'p_value': p_value}

# Test specific transitions
test_transition_difference(
    P_trad=model.P_traditional,
    P_token=model.P_tokenized,
    from_regime='panic',
    to_regime='panic',
    n_trad=67,
    n_token=16
)
```

---

## 4. For Your Thesis

### 4.1 Minimum Viable Validation

**Priority 1 (Must have):**
1. Out-of-sample accuracy (train/test split)
2. Historical event validation (2008, 2020 crises)
3. Bootstrap CIs for key estimates
4. Robustness to regime definition

**Time Required:** 3-4 days

**Implementation:**
```python
# Complete validation script for thesis

# 1. Train/test split
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

model.fit(train_data)
train_acc = model.evaluate(train_data)
test_acc = model.evaluate(test_data)

print(f"In-sample accuracy: {train_acc:.1%}")
print(f"Out-of-sample accuracy: {test_acc:.1%}")

# 2. Historical events
crisis_accuracy = validate_regime_inference(model, KNOWN_CRISES)

# 3. Bootstrap CIs
key_metrics = {
    'calm_to_neutral': lambda m: calm_to_neutral_prob(m),
    'panic_persistence': lambda m: panic_to_panic_prob(m),
    # ... add more
}

for name, metric_func in key_metrics.items():
    ci = bootstrap_confidence_interval(model, train_data, metric_func)
    print(f"\n{name}: {ci['estimate']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")

# 4. Robustness
for threshold_set in [base, low, high]:
    regimes = infer_regimes(data, thresholds=threshold_set)
    consistency = calculate_consistency(regimes, base_regimes)
    print(f"Threshold sensitivity: κ = {consistency:.2f}")
```

---

### 4.2 Reporting in Thesis

**Table: Model Validation Results**
```
Metric                          Value       95% CI
---------------------------------------------------
Out-of-sample accuracy         72.3%    [68.1%, 76.2%]
Crisis detection rate          88.9%    [8/9 crises]
Cross-validation score        0.714    [0.683, 0.745]

Transition Probabilities (Traditional):
  Calm → Neutral              13.9%    [10.7%, 17.5%]
  Panic → Volatile            75.0%    [62.3%, 85.1%]

Transition Probabilities (Tokenized):
  Calm → Neutral              17.4%    [11.2%, 25.8%]
  Panic → Panic               90.0%    [71.4%, 98.2%]**

Note: Confidence intervals via bootstrap (n=1,000).
Wider CIs for tokenized reflect smaller sample size (n=16 vs n=67).
```

---

## 5. Complete Validation Checklist

**Before thesis submission:**

**Statistical Validation:**
- [ ] Train/test split reported
- [ ] Cross-validation performed
- [ ] Bootstrap CIs calculated
- [ ] Significance tests conducted
- [ ] Effect sizes reported
- [ ] Multiple comparison correction (if applicable)

**Economic Validation:**
- [ ] Historical events tested
- [ ] Economic theory consistency discussed
- [ ] Alternative explanations considered
- [ ] Policy implications analyzed

**Robustness Checks:**
- [ ] Sensitive to regime thresholds?
- [ ] Sensitive to time period?
- [ ] Sensitive to interpolation method?
- [ ] Consistent across subsamples?

**Reporting:**
- [ ] All tests pre-specified
- [ ] Negative results reported
- [ ] Limitations acknowledged
- [ ] Reproducibility enabled

---

## 6. References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. CRC Press.
- Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.

---

**For implementation examples, see:** [Future Improvements](02_FUTURE_IMPROVEMENTS.md) - Section 1.3
