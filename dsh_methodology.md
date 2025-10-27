# Methodological Documentation

## Ancient Text Numerical Analysis Framework - DSH Edition

**Author**: Ahmed Benseddik  
**Version**: 4.0-DSH  
**Last Updated**: 2025-10-26

---

## Table of Contents

1. [Research Questions](#research-questions)
2. [Theoretical Framework](#theoretical-framework)
3. [Data Collection](#data-collection)
4. [Statistical Methods](#statistical-methods)
5. [Validation Strategy](#validation-strategy)
6. [Ethical Considerations](#ethical-considerations)
7. [Limitations](#limitations)
8. [References](#references)

---

## 1. Research Questions

### Primary Questions

**RQ1**: Do numerical values derived from ancient text segments exhibit patterns that deviate from random distribution?

**RQ2**: Are specific divisors (7, 12, 26, 30, 60) enriched beyond chance expectations in gematria distributions?

**RQ3**: How robust are observed patterns across different analytical parameters (window size, sampling strategy)?

### Hypotheses

**H₀ (Null Hypothesis)**: Numerical patterns in ancient texts follow random uniform distribution.

**H₁ (Alternative Hypothesis)**: Certain numerical patterns show statistically significant enrichment.

---

## 2. Theoretical Framework

### 2.1 Cultural Numerical Systems

#### Hebrew Gematria
- **Origin**: Ancient Jewish interpretive tradition
- **System**: Each Hebrew letter assigned numerical value (א=1, ב=2, ..., ת=400)
- **Variants**: Standard, Atbash (cipher), Albam (alternative encoding)
- **Cultural Context**: Used in biblical exegesis, Kabbalah, rabbinic literature

#### Greek Isopsephy
- **Origin**: Hellenistic period (3rd century BCE onward)
- **System**: Greek alphabet with numerical values (α=1, β=2, ..., ω=800)
- **Usage**: Found in New Testament manuscripts, ancient inscriptions

#### Arabic Abjad
- **Origin**: Pre-Islamic Arabian tradition
- **System**: Arabic letters with assigned values (ا=1, ب=2, ..., غ=1000)
- **Usage**: Islamic mysticism, poetry, chronograms

### 2.2 Methodological Approach

This framework employs **computational text analysis** combined with **rigorous statistical inference** to:

1. Extract numerical patterns systematically
2. Test hypotheses using multiple statistical paradigms
3. Validate results through sensitivity analysis
4. Contextualize findings within cultural frameworks

**Key Principle**: Numerical patterns are descriptive findings that require cultural-historical interpretation, not evidence of intentionality.

---

## 3. Data Collection

### 3.1 Text Preparation

#### Input Requirements
- **Format**: UTF-8 encoded plain text
- **Script**: Hebrew (primary), Greek, or Arabic
- **Preprocessing**: 
  - Normalize final letter forms (Hebrew: ךםןףץ → כמנפצ)
  - Remove non-alphabetic characters
  - Preserve original character order

#### Sampling Strategy

**Window-based Extraction**:
```
Text: [א][ב][ג][ד][ה][ו][ז][ח][ט][י]...
Windows (size=3, stride=1): 
  - אבג
  - בגד
  - גדה
  ...
```

**Parameters**:
- **Window size (w)**: 3, 5, 7, 10 characters
- **Stride (s)**: 1, 5, 10 (controls overlap)
- **Sample size (n)**: Minimum 100 for adequate power

### 3.2 Quality Control

- **Encoding validation**: Verify UTF-8 integrity
- **Character set validation**: Confirm script consistency
- **Completeness check**: Flag missing sections
- **Duplicate detection**: Identify repeated passages

---

## 4. Statistical Methods

### 4.1 Frequentist Approach

#### Binomial Test
**Purpose**: Test if proportion of multiples differs from expectation

**Model**:
```
H₀: p = 1/d (where d = divisor)
H₁: p > 1/d (one-tailed) or p ≠ 1/d (two-tailed)

Test statistic: k (observed multiples)
Distribution: Binomial(n, p)
```

**Implementation**:
```python
from scipy.stats import binomtest
result = binomtest(k, n, p=1/divisor, alternative='greater')
```

**Effect Size** (Cohen's h):
```
h = 2 × (arcsin(√p₁) - arcsin(√p₂))

Interpretation:
  |h| < 0.2: negligible
  0.2 ≤ |h| < 0.5: small
  0.5 ≤ |h| < 0.8: medium
  |h| ≥ 0.8: large
```

#### Multiple Testing Corrections

**Bonferroni Correction**:
```
α_adjusted = α / m
where m = number of tests
```

**Šidák Correction** (more powerful):
```
α_adjusted = 1 - (1 - α)^(1/m)
```

**Benjamini-Hochberg FDR**:
```
1. Order p-values: p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍ₘ₎
2. Find largest i where p₍ᵢ₎ ≤ (i/m) × α
3. Reject H₀ for all tests 1, ..., i
```

### 4.2 Bayesian Approach

#### Hierarchical Model

**Likelihood**:
```
k ~ Binomial(n, p)
```

**Priors**:
```
Null model:
  p ~ Beta(1, 1)  [Uniform prior]

Enrichment model:
  α ~ Exponential(1)
  β ~ Exponential(1)
  p ~ Beta(α, β)  [Informative prior centered near 1/d]
```

**Model Comparison**:
```
WAIC (Widely Applicable Information Criterion):
  WAIC = -2 × (lppd - p_WAIC)
  
  where:
    lppd = log pointwise predictive density
    p_WAIC = effective number of parameters

Interpretation:
  ΔWAIC < 2: Models equivalent
  2 ≤ ΔWAIC < 6: Weak evidence
  ΔWAIC ≥ 6: Strong evidence
```

**Bayes Factor** (approximation):
```
BF ≈ exp((WAIC_null - WAIC_alternative) / 2)

Interpretation (Jeffreys' scale):
  1-3: Weak evidence
  3-10: Moderate evidence
  10-30: Strong evidence
  >30: Very strong evidence
```

### 4.3 Non-Parametric Methods

#### Permutation Test

**Algorithm**:
```
1. Compute observed statistic: T_obs = f(data)
2. For i = 1 to n_permutations:
     a. Shuffle data randomly
     b. Compute T_perm[i] = f(shuffled_data)
3. p-value = mean(|T_perm| ≥ |T_obs|)
```

**Advantages**:
- No distributional assumptions
- Exact p-values for finite samples
- Robust to outliers

**Disadvantages**:
- Computationally intensive
- Requires exchangeability assumption

### 4.4 Power Analysis

**Purpose**: Determine if sample size is adequate to detect effects

**Formula** (proportion test):
```
n = (z_α/2 + z_β)² × [p₀(1-p₀) + p₁(1-p₁)] / (p₁ - p₀)²

where:
  z_α/2: critical value for significance level α
  z_β: critical value for power (1-β)
  p₀: null proportion
  p₁: alternative proportion
```

**Implementation**:
```python
def compute_power(n, effect_size, alpha=0.05):
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = (effect_size * np.sqrt(n)) - z_alpha
    power = norm.cdf(z_beta)
    return power
```

**Recommendations**:
- Target power ≥ 0.80
- For small effects (h=0.2), need n ≥ 400
- For medium effects (h=0.5), need n ≥ 64

---

## 5. Validation Strategy

### 5.1 Internal Validation

#### Distribution Testing
- **Shapiro-Wilk test**: Normality assumption
- **Anderson-Darling test**: Goodness-of-fit
- **Kolmogorov-Smirnov test**: Compare to theoretical distribution
- **Q-Q plots**: Visual normality assessment

#### Assumption Checking
- **Independence**: No autocorrelation (Durbin-Watson test)
- **Sample size**: Adequate for power ≥ 0.80
- **Effect magnitude**: Practical significance vs. statistical

### 5.2 Sensitivity Analysis

**Parameters to Vary**:

1. **Window size** (w): 3, 5, 7, 10
2. **Stride** (s): 1, 5, 10, 20
3. **Divisors** (d): 7, 12, 26, 30, 60
4. **Significance level** (α): 0.01, 0.05, 0.10

**Metrics**:
- **Coefficient of variation** (CV): std(results) / mean(results)
  - CV < 0.3: Robust
  - 0.3 ≤ CV < 0.5: Moderate sensitivity
  - CV ≥ 0.5: High sensitivity

- **Rank correlation**: Spearman's ρ across parameter sets
  - ρ > 0.7: Consistent rankings
  - 0.4 < ρ ≤ 0.7: Moderate consistency
  - ρ ≤ 0.4: Inconsistent

### 5.3 Cross-Validation

**k-Fold Cross-Validation**:
```
1. Split data into k equal folds
2. For each fold i:
     - Train on k-1 folds
     - Test on fold i
     - Record performance
3. Average performance across folds
```

**Stratified Sampling**:
- Ensure representative distribution in each fold
- Preserve class proportions (for categorical outcomes)

---

## 6. Ethical Considerations

### 6.1 Cultural Sensitivity

**Principles**:
1. **Respect tradition**: Numerical systems are cultural artifacts
2. **Avoid reductionism**: Quantitative analysis doesn't capture full meaning
3. **Acknowledge limitations**: Statistical patterns ≠ intentionality
4. **Multiple perspectives**: Different communities may interpret differently

**Community Engagement**:
- Consult traditional scholars
- Present findings with appropriate caveats
- Acknowledge cultural specificity of methods

### 6.2 Interpretation Guidelines

**What Results CAN Show**:
- ✓ Descriptive patterns in numerical distributions
- ✓ Statistical deviations from randomness
- ✓ Comparative analysis across systems

**What Results CANNOT Show**:
- ✗ Authorial intent or consciousness
- ✗ "Hidden messages" or prophecies
- ✗ Superiority of one tradition over another
- ✗ Definitive meanings without context

### 6.3 Transparency Requirements

**Mandatory Disclosures**:
1. Complete methodology with code
2. All parameter choices and justifications
3. Negative results (null findings)
4. Limitations and potential biases
5. Funding sources and conflicts of interest

---

## 7. Limitations

### 7.1 Methodological Limitations

1. **Text Segmentation**: Window-based extraction is arbitrary
2. **Multiple Testing**: Family-wise error rate increases with tests
3. **P-hacking Risk**: Iterative testing can inflate false positives
4. **Publication Bias**: Tendency to report only significant findings

### 7.2 Interpretive Limitations

1. **Correlation ≠ Causation**: Patterns don't imply design
2. **Cultural Context**: Numerical systems are culture-specific
3. **Historical Distance**: Ancient contexts differ from modern
4. **Multiplicity**: Many valid interpretations may coexist

### 7.3 Technical Limitations

1. **Computational Constraints**: Large-scale analyses are resource-intensive
2. **Statistical Power**: Small effects require large samples
3. **Convergence Issues**: Bayesian MCMC may not converge for complex models
4. **Software Dependencies**: Results depend on library implementations

---

## 8. References

### Statistical Methods

1. **Efron, B., & Tibshirani, R. J. (1994)**. *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

2. **Gelman, A., Carlin, J. B., Stern, H. S., et al. (2013)**. *Bayesian Data Analysis* (3rd ed.). CRC Press.

3. **Good, P. I. (2005)**. *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.

4. **Benjamini, Y., & Hochberg, Y. (1995)**. Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.

5. **Cohen, J. (1988)**. *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Routledge.

### Cultural Systems

6. **Ifrah, G. (2000)**. *The Universal History of Numbers: From Prehistory to the Invention of the Computer*. Wiley.

7. **Scholem, G. (1974)**. *Kabbalah*. Keter Publishing House.

8. **Blech, B. (2004)**. *The Complete Idiot's Guide to Jewish Culture*. Alpha Books.

### Digital Humanities

9. **Schöch, C. (2013)**. Big? Smart? Clean? Messy? Data in the Humanities. *Journal of Digital Humanities*, 2(3), 2-13.

10. **Jockers, M. L. (2013)**. *Macroanalysis: Digital Methods and Literary History*. University of Illinois Press.

11. **Underwood, T. (2019)**. *Distant Horizons: Digital Evidence and Literary Change*. University of Chicago Press.

### Reproducibility

12. **Stodden, V., Leisch, F., & Peng, R. D. (2014)**. *Implementing Reproducible Research*. CRC Press.

13. **Marwick, B., Boettiger, C., & Mullen, L. (2018)**. Packaging data analytical work reproducibly using R (and friends). *The American Statistician*, 72(1), 80-88.

---

## Appendix A: Glossary

**Binomial Test**: Statistical test for proportion equality  
**Bonferroni Correction**: Conservative multiple testing adjustment  
**Cohen's h**: Effect size for proportions  
**FDR**: False Discovery Rate  
**Gematria**: Hebrew numerical system  
**HDI**: Highest Density Interval (Bayesian credible interval)  
**Isopsephy**: Greek numerical system  
**MCMC**: Markov Chain Monte Carlo (Bayesian sampling)  
**Power**: Probability of detecting true effect  
**WAIC**: Widely Applicable Information Criterion  

---

## Appendix B: Checklist for DSH Reviewers

### Methodological Rigor
- [ ] Research questions clearly stated
- [ ] Hypotheses explicitly formulated
- [ ] Statistical methods appropriate for data
- [ ] Multiple testing corrections applied
- [ ] Power analysis conducted
- [ ] Sensitivity analysis performed

### Reproducibility
- [ ] Code publicly available
- [ ] Data accessible (or restrictions explained)
- [ ] Random seeds documented
- [ ] Environment specifications provided
- [ ] Results independently verifiable

### Ethical Considerations
- [ ] Cultural sensitivity addressed
- [ ] Interpretation limitations acknowledged
- [ ] Community perspectives considered
- [ ] Conflicts of interest disclosed

### Documentation Quality
- [ ] Methods clearly described
- [ ] Results transparently reported
- [ ] Figures publication-quality
- [ ] Tables properly formatted
- [ ] References complete and accurate

---

*Document Version: 1.0*  
*Last Reviewed: 2025-10-26*  
*Next Review: 2026-10-26*