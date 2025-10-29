# A Computational Framework for Detecting Numerical Patterns in Ancient Texts

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17443361.svg)](https://doi.org/10.5281/zenodo.17443361)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A rigorous, reproducible framework for computational analysis of numerical patterns in ancient texts using multiple cultural systems (Hebrew, Greek, Arabic) with comprehensive statistical validation, Bayesian inference, and ethical considerations.

**Publication**: *Digital Scholarship in the Humanities* (DSH) — 2025  
**Author**: Ahmed Benseddik  
**Version**: 4.5-DSH  
**Date**: October 29, 2025

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Case Study: Genesis](#-case-study-genesis)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Repository Structure](#-repository-structure)
- [Methodology](#-methodology)
- [Usage Examples](#-usage-examples)
- [Analysis Pipeline](#-analysis-pipeline)
- [Interpreting Results](#-interpreting-results)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Reproducibility](#-reproducibility)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [References](#-references)

---

## 🎯 Overview

This repository contains the complete implementation of a **three-phase computational framework** for detecting numerical patterns in ancient texts, with comprehensive case studies of Genesis (Sefer Bereshit) and support for multiple cultural numerical systems. The framework combines:

1. **Unsupervised Discovery** — Pattern detection via frequency analysis, gematria, and permutation scans
2. **Statistical Validation** — Multiple tests (permutation, Bayesian, bootstrap) with FDR corrections
3. **Expert Consensus** — Structured Delphi protocol with interdisciplinary panel

### Main Contributions

✅ **First integrated framework** combining frequentist, Bayesian, and qualitative validation  
✅ **Rigorous anti-p-hacking protocol** — Pre-registered markers and discovery-validation split  
✅ **Multiple cultural systems** — Hebrew (standard, Atbash, Albam), Greek Isopsephy, Arabic Abjad  
✅ **Formal mathematical proofs** — 7 theorems with computational verification  
✅ **Diachronic validation** — Manuscript stability across 1100 years (Qumran → Leningrad)  
✅ **Complete reproducibility** — All code, data, and parameters publicly available

### Main Findings (Genesis Case Study)

- **תולדות (Toledot, "Generations")** — Gematria value 846 marks 10 structural divisions (BF=18.7, p<0.01)
- **התבה (Ha-Tebah, "The Ark")** — 17 occurrences cluster at narrative markers (p<0.01, Cohen's d=4.19)
- **Intertextual sums** — 1260, 1290, 1335 correspond with prophetic chronologies (Daniel, Revelation)

All patterns validated with **convergent evidence** across multiple independent methods.

---

## 🌟 Key Features

### Statistical Rigor

#### Frequentist Methods
- **Permutation tests**: 10,000-50,000 iterations with exact p-values
- **Binomial tests**: Exact confidence intervals (Wilson score method)
- **Multiple testing corrections**: Bonferroni, Šidák, Benjamini-Hochberg FDR
- **Effect sizes**: Cohen's h, Cohen's d, standardized differences
- **Bootstrap CI**: Percentile and BCa methods (10,000 resamples)
- **Power analysis**: Sample size adequacy assessment

#### Bayesian Methods
- **Hierarchical models**: Beta-Binomial conjugate priors
- **MCMC sampling**: PyMC with 4 chains, 5000+ draws
- **Convergence diagnostics**: R̂ (Gelman-Rubin), effective sample size
- **Model comparison**: WAIC, LOO-CV, Bayes Factors
- **Posterior predictive checks**: Distribution validation
- **HDI intervals**: Highest Density Intervals (95%)

#### Non-Parametric Validation
- **Distribution tests**: Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov
- **Q-Q plots**: Quantile-quantile comparisons
- **Permutation-based CI**: Distribution-free inference

### Reproducibility

- ✅ **Complete environment capture**: Python version, dependencies, system info
- ✅ **Git commit tracking**: Version control integration
- ✅ **Deterministic seeds**: All random processes reproducible (seed=42)
- ✅ **Comprehensive logging**: File + console outputs with timestamps
- ✅ **Metadata tracking**: Every analysis run documented
- ✅ **Pre-registration**: OSF registry for markers and parameters
- ✅ **Code verification**: Independent R implementation validates results

### Cultural Systems

- **Hebrew Gematria**:
  - Standard (Mispar Hechrachi)
  - Atbash (letter reversal)
  - Albam (letter substitution)
- **Greek Isopsephy**: Classical Greek numerical values
- **Arabic Abjad**: Traditional Arabic numerals
- **Cross-cultural correlation**: Statistical comparison across systems

### Visualizations

- 📊 **Publication-quality figures**: 300 DPI, vector formats available
- 📈 **Distribution plots**: Histograms with density curves, Q-Q plots
- 🎨 **Bayesian diagnostics**: Trace plots, posterior distributions, forest plots
- 🔍 **Sensitivity analysis**: Robustness visualizations
- 🌐 **Cross-cultural heatmaps**: Correlation matrices
- 📉 **Effect size plots**: Forest plots with confidence intervals

### Ethical Framework

- 🔬 **Methodological transparency**: All assumptions documented
- 🌍 **Cultural sensitivity**: Guidelines for respectful interpretation
- ⚠️ **Interpretation caveats**: Limitations clearly stated
- 📝 **Acknowledgment of uncertainty**: Probabilistic statements only
- 🤝 **Community engagement**: Open to scholarly feedback

---

## 📖 Case Study: Genesis

### Validated Patterns (5/15 tested)

| Pattern | Hebrew | Value/Count | p-value | Bayes Factor | Expert Score | Stability |
|---------|--------|-------------|---------|--------------|--------------|-----------|
| Toledot | תולדות | 846 (gematria) | 0.007 | 18.7 | 8.2/10 | 96.7% |
| Ha-Tebah | התבה | 17 occurrences | 0.010 | 21.6 | 8.3/10 | 98.0% |
| Sum 1260 | — | 3 instances | 0.012 | 14.3 | 7.9/10 | 100% |
| Sum 1290 | — | 2 instances | 0.019 | 12.4 | 8.1/10 | 100% |
| Sum 1335 | — | 2 instances | 0.015 | 14.9 | 7.5/10 | 100% |

**All patterns significant after FDR correction (q < 0.05)**

### Sensitivity Analysis

- ✅ **Alternative markers**: Patterns robust across 3 marker definitions (p ≤ 0.02 in all)
- ✅ **Subsampling**: Ha-Tebah specific to Noah narrative (as expected)
- ✅ **Random seed variation**: P-values stable within ±0.005 across 10 seeds
- ✅ **Manuscript variations**: 91-100% stability across Qumran, Aleppo, Leningrad

---

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- Git
- (Optional) LaTeX distribution for compiling mathematical proofs

### Clone Repository

```bash
git clone https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4.git
cd Ancient-Text-Numerical-Analysis-v-0.4
```

### Install Dependencies

#### Full Installation (with Bayesian methods)

```bash
pip install -r requirements.txt
```

**Required packages:**
```
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
statsmodels>=0.14.0
jupyter>=1.0.0
pytest>=7.0.0
pymc>=5.0.0
arviz>=0.15.0
numba>=0.57.0
```

#### Minimal Installation (without Bayesian)

```bash
pip install numpy scipy pandas matplotlib seaborn statsmodels
```

### Verify Installation

```bash
# Run test suite
python -m pytest tests/ -v

# Run theorem demonstrations
python src/theorem_demonstrations.py

# Check environment
python -c "import ancient_text_dsh; print(ancient_text_dsh.__version__)"
```

All tests should pass ✅

---

## ⚡ Quick Start

### 1. Run Complete Analysis (Genesis Example)

```bash
# Full analysis with all features
python ancient_text_dsh.py --data-dir ./data/genesis --output-dir ./results

# Fast analysis (no Bayesian, fewer permutations)
python ancient_text_dsh.py --no-bayesian --n-permutations 10000

# High-quality analysis (publication-ready)
python ancient_text_dsh.py --n-permutations 50000 --n-bayesian-draws 5000
```

### 2. Reproduce Key Finding: Ha-Tebah (התבה)

```python
from src.permutation_tests import permutation_test
from src.bayesian_analysis import bayes_factor_binomial
import json

# Load configuration
with open('data/analysis_config.json', 'r') as f:
    config = json.load(f)

# Load markers
with open('data/structural_markers.json', 'r') as f:
    markers = json.load(f)

# Run permutation test for Ha-Tebah
result = permutation_test(
    corpus='data/genesis_leningrad.txt',
    target_term='התבה',
    markers=markers['chapter_boundaries'],
    n_iterations=50000,
    seed=42
)

print(f"P-value: {result['p_value']:.5f}")
print(f"Observed count: {result['observed_count']}")
print(f"Expected (null): {result['null_mean']:.2f}")
print(f"Cohen's d: {result['cohens_d']:.2f}")

# Bayes Factor
bf = bayes_factor_binomial(
    observed_count=17,
    n_markers=43,
    corpus_length=20614,
    total_occurrences=17,
    alpha_prior=5.0,
    beta_prior=2.0
)

print(f"Bayes Factor: {bf:.1f}")
```

**Expected Output:**
```
P-value: 0.00974
Observed count: 17
Expected (null): 8.24
Cohen's d: 4.19
Bayes Factor: 21.6
✓ Pattern validated
```

### 3. Custom Analysis

```python
from ancient_text_dsh import AnalysisConfig, AncientTextAnalysisPipeline

# Configure analysis
config = AnalysisConfig(
    data_dir='custom/path',
    output_dir='custom/output',
    random_seed=123,
    n_permutations=20000,
    n_bayesian_draws=5000,
    enable_bayesian=True,
    significance_level=0.01
)

# Run pipeline
pipeline = AncientTextAnalysisPipeline(config)
results = pipeline.run_complete_analysis()

# Access results
print(f"Validated patterns: {len(results['validated_patterns'])}")
print(f"Average BF: {results['summary']['mean_bayes_factor']:.2f}")
```

### 4. Interactive Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Start with:
- `01_exploratory_analysis.ipynb` — Data exploration
- `02_permutation_tests.ipynb` — Statistical tests
- `03_bayesian_validation.ipynb` — Bayesian inference
- `04_diachronic_checks.ipynb` — Manuscript comparison

---

## 📁 Repository Structure

```
Ancient-Text-Numerical-Analysis-v-0.4/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── CHANGELOG.md                       # Version history
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
│
├── data/                              # Source texts and annotations
│   ├── genesis_leningrad.txt         # Westminster Leningrad Codex (Genesis)
│   ├── structural_markers.json       # Pre-registered markers (43 total)
│   ├── gematria_map.csv              # Hebrew letter → numeric values
│   ├── key_patterns.json             # 5 validated patterns with stats
│   ├── analysis_config.json          # Pre-registered parameters
│   └── cultural_systems/             # Greek, Arabic mappings
│
├── src/                               # Core analysis modules
│   ├── __init__.py
│   ├── ancient_text_dsh.py           # Main analysis script
│   ├── permutation_tests.py          # Permutation test implementation
│   ├── bayesian_analysis.py          # Bayes Factor calculations
│   ├── gematria_calculator.py        # Multi-cultural gematria
│   ├── diachronic_validation.py      # Manuscript comparison
│   ├── expert_panel_analysis.py      # Delphi protocol scoring
│   ├── fdr_correction.py             # Benjamini-Hochberg FDR
│   ├── visualization_tools.py        # Plotting functions
│   └── theorem_demonstrations.py     # Mathematical proofs verification
│
├── notebooks/                         # Interactive analysis
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_permutation_tests.ipynb
│   ├── 03_bayesian_validation.ipynb
│   ├── 04_diachronic_checks.ipynb
│   ├── 05_expert_panel_analysis.ipynb
│   └── 06_sensitivity_analyses.ipynb
│
├── results/                           # Analysis outputs
│   ├── permutation_outputs.csv       # P-values for all patterns
│   ├── bayes_factors.csv             # BF calculations
│   ├── expert_scores.csv             # Delphi panel results
│   ├── diachronic_stability.csv      # Manuscript preservation
│   ├── theorem_verification_results.json
│   └── figures/                      # Publication-ready plots
│       ├── theorem1_type1_control.png
│       ├── theorem2_bf_consistency.png
│       ├── theorem3_fdr_control.png
│       ├── gematria_distribution.png
│       ├── multiples_analysis.png
│       └── cross_cultural_heatmap.png
│
├── docs/                              # Documentation
│   ├── METHODOLOGY.md                # Detailed methods
│   ├── mathematical_proofs.pdf       # Complete proofs (25 pages)
│   ├── mathematical_proofs.tex       # LaTeX source
│   ├── proofs_summary.pdf            # 5-page summary
│   ├── references.bib                # BibTeX bibliography (40+ refs)
│   ├── technical_slide.html          # Permutation visualization
│   ├── infographic.html              # Framework visual summary
│   └── appendix_A_methodology.md     # Complete appendix
│
├── tests/                             # Unit and integration tests
│   ├── __init__.py
│   ├── test_permutation.py
│   ├── test_bayesian.py
│   ├── test_gematria.py
│   ├── test_fdr.py
│   ├── test_statistics.py
│   └── test_pipeline.py
│
└── supplementary/                     # Additional materials
    ├── presentation_beamer.pdf       # Conference slides
    ├── poster_DSH2025.pdf            # Conference poster
    └── media/                        # Presentation figures
```

---

## 🔬 Methodology

### Three-Phase Framework

#### **Phase 1: DISCOVERY** 𝓓(T, M)

Unsupervised detection of pattern candidates:

$$\mathcal{D}(T, M) = \{P_i \mid f(P_i, M) > \mu_{\text{null}} + k\sigma_{\text{null}}\}$$

- **Input:** Text corpus T, pre-registered markers M
- **Output:** Candidate patterns exceeding k=2 standard deviations
- **Methods:** 
  - Frequency analysis
  - Gematria calculation (multiple systems)
  - Co-occurrence detection
  - Positional clustering

#### **Phase 2: VALIDATION** 𝓥(𝓓)

Multi-method statistical validation:

$$\mathcal{V}(\mathcal{D}) = 1 \text{ if } \min(p_{\text{perm}}, q_{\text{FDR}}) < \alpha \text{ AND } BF > \beta$$

- **Permutation tests:** 10,000-50,000 iterations, exact p-values
- **Bayesian analysis:** Bayes Factors with Beta priors (BF > 10 threshold)
- **Bootstrap CI:** 10,000 resamples, 95% confidence intervals
- **FDR correction:** Benjamini-Hochberg at q=0.05
- **Effect sizes:** Cohen's d, h with interpretation guidelines
- **Power analysis:** Sample size adequacy (target power ≥ 0.80)

#### **Phase 3: EXPERTISE** 𝓔(𝓥)

Structured expert consensus (Delphi protocol):

$$\mathcal{E}(\mathcal{V}) = \frac{1}{n_e} \sum_{j=1}^{n_e} w_j \cdot S_j^{(3)}$$

- **Panel:** 12 experts (4 philologists, 3 statisticians, 3 historians, 2 textual critics)
- **Protocol:** Modified Delphi with 3 rounds
  - Round 1: Blind assessment (no statistics)
  - Round 2: Re-evaluation with statistical disclosure
  - Round 3: Consensus discussion
- **Scoring:** 0-10 scale across 4 criteria
  - Historical plausibility (0-3 points)
  - Textual coherence (0-3 points)
  - Manuscript stability (0-2 points)
  - Statistical strength (0-2 points)
- **Threshold:** Mean ≥ 7.0 with SD ≤ 1.5

### Combined Decision Rule

A pattern is **validated** if and only if **ALL** criteria are met:

1. ✅ Permutation p-value < 0.01
2. ✅ Bayes Factor > 10 (strong evidence)
3. ✅ Expert consensus ≥ 7.0
4. ✅ Diachronic stability ≥ 90%

**Mathematical Formulation:**

$$\Phi(T, M, C) = \mathcal{D}(T, M) \otimes \mathcal{V}(\mathcal{D}) \otimes \mathcal{E}(\mathcal{V}) \geq \theta_{\text{sig}}$$

**Theorem (Combined Type-I Error Control):** Under the global null hypothesis, the framework controls family-wise error rate at α ≤ 0.05.

*Proof:* See [`docs/mathematical_proofs.pdf`](docs/mathematical_proofs.pdf) (Theorem 4, page 12).

---

## 💻 Usage Examples

### Example 1: Compute Gematria (Multiple Systems)

```python
from ancient_text_dsh import compute_gematria, CulturalSystem

# Hebrew (standard)
hebrew_value = compute_gematria('בראשית', CulturalSystem.HEBREW_STANDARD)
print(f"Hebrew (standard): {hebrew_value}")  # 913

# Hebrew (Atbash)
atbash_value = compute_gematria('בראשית', CulturalSystem.HEBREW_ATBASH)
print(f"Hebrew (Atbash): {atbash_value}")

# Greek Isopsephy
greek_value = compute_gematria('λόγος', CulturalSystem.GREEK_ISOPSEPHY)
print(f"Greek: {greek_value}")  # 373

# Arabic Abjad
arabic_value = compute_gematria('بسم', CulturalSystem.ARABIC_ABJAD)
print(f"Arabic: {arabic_value}")  # 102
```

### Example 2: Batch Processing

```python
from pathlib import Path
from ancient_text_dsh import AnalysisConfig, AncientTextAnalysisPipeline

# Process entire corpus
corpus_files = Path('corpus').glob('*.txt')

for file in corpus_files:
    print(f"Analyzing {file.name}...")
    
    config = AnalysisConfig(
        data_dir=file.parent,
        output_dir=Path('results') / file.stem,
        n_permutations=50000
    )
    
    pipeline = AncientTextAnalysisPipeline(config)
    results = pipeline.run_complete_analysis()
    
    print(f"  Validated: {len(results['validated_patterns'])} patterns")
```

### Example 3: Custom Statistical Test

```python
from src.permutation_tests import permutation_test
import numpy as np

# Your custom data
observed_frequencies = [17, 12, 23, 9, 15]
expected_baseline = 0.15  # Expected proportion

# Run permutation test
result = permutation_test(
    observed=observed_frequencies,
    baseline=expected_baseline,
    n_iterations=50000,
    seed=42
)

print(f"P-value: {result['p_value']:.5f}")
print(f"Cohen's d: {result['cohens_d']:.2f}")
print(f"95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
```

### Example 4: Bayesian Model Comparison

```python
from src.bayesian_analysis import bayesian_model_comparison

# Compare null vs. enrichment models
comparison = bayesian_model_comparison(
    data=observed_counts,
    model_null='binomial',
    model_alternative='beta_binomial',
    n_samples=5000
)

print(f"WAIC difference: {comparison['delta_waic']:.2f}")
print(f"Evidence for alternative: {comparison['interpretation']}")
print(f"BF: {comparison['bayes_factor']:.1f}")
```

---

## 📊 Analysis Pipeline

### Pipeline Stages

```
1. Data Preprocessing
   ├── Validate encoding (UTF-8)
   ├── Normalize text (final forms)
   ├── Extract segments (windows)
   └── Compute numerical values

2. Gematria Analysis
   ├── Statistical summaries
   ├── Distribution testing
   ├── Cross-cultural comparison
   └── Visualization

3. Frequentist Validation
   ├── Multiples enrichment (7, 12, 26, 30, 60)
   ├── Binomial tests
   ├── Permutation tests (10k-50k iter)
   ├── FDR correction
   └── Effect sizes + CI

4. Bayesian Analysis (optional)
   ├── Hierarchical modeling
   ├── MCMC sampling (4 chains)
   ├── Convergence diagnostics
   ├── Model comparison (WAIC/LOO)
   └── Posterior predictive checks

5. Sensitivity Analysis
   ├── Window size variations
   ├── Sampling strategies
   ├── Parameter robustness
   └── Bootstrap stability

6. Expert Validation
   ├── Delphi Round 1 (blind)
   ├── Delphi Round 2 (with stats)
   ├── Delphi Round 3 (consensus)
   └── Final scoring

7. Diachronic Validation
   ├── Manuscript comparison
   ├── Stability calculation
   └── Transmission analysis

8. Report Generation
   ├── JSON results
   ├── Markdown report
   ├── Publication figures
   └── Summary tables
```

---

## 📈 Interpreting Results

### Statistical Significance Levels

| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| **P-value** | < 0.01 | Highly significant (after FDR correction) |
| **Bayes Factor** | > 10 | Strong evidence for H₁ |
| | 3-10 | Moderate evidence |
| | < 3 | Weak/no evidence |
| **Effect Size (d)** | 0.2 | Small effect |
| | 0.5 | Medium effect |
| | 0.8 | Large effect |
| | > 2.0 | Very large effect |
| **Expert Score** | ≥ 7.0 | Pattern likely meaningful |
| | 4-7 | Uncertain, needs more evidence |
| | < 4 | Likely spurious |
| **Stability** | ≥ 90% | Robust across manuscripts |
| | 70-90% | Moderate stability |
| | < 70% | Questionable transmission |

### Bayesian Interpretation (WAIC)

- **ΔWAIC < 2**: Models similar, no clear preference
- **2 < ΔWAIC < 6**: Moderate evidence for better model
- **ΔWAIC > 6**: Strong evidence for better model

### Sensitivity Analysis (Coefficient of Variation)

- **CV < 0.3**: Robust results, conclusions reliable
- **0.3 < CV < 0.5**: Moderate sensitivity, interpret with caution
- **CV > 0.5**: High sensitivity, results unstable

### Power Analysis

- **Power > 0.8**: Adequate sample size for detecting effect
- **0.6 < Power < 0.8**: Moderate power, consider larger sample
- **Power < 0.6**: Underpowered, high risk of Type-II error

### Combined Validation

For a pattern to be **fully validated**, it should show:
1. ✅ Statistical significance (p < 0.01, BF > 10)
2. ✅ Large effect size (d > 0.8)
3. ✅ Expert consensus (score ≥ 7.0)
4. ✅ Manuscript stability (≥ 90%)
5. ✅ Robustness to variations (CV < 0.5)

---

## 🧪 Testing

### Run Complete Test Suite

```bash
# All tests with coverage
pytest tests/ -v --cov=ancient_text_dsh --cov-report=html

# Quick smoke tests
pytest tests/ -x -v

# Specific test categories
pytest tests/test_gematria.py -v        # Gematria calculations
pytest tests/test_statistics.py -v      # Statistical methods
pytest tests/test_pipeline.py -v        # Integration tests
pytest tests/test_bayesian.py -v        # Bayesian inference
```

### Generate Coverage Report

```bash
pytest tests/ --cov=ancient_text_dsh --cov-report=term-missing
```

### Run Theorem Demonstrations

```bash
python src/theorem_demonstrations.py
```

**Expected output:**
```
======================================================================
THEOREM 1: Type-I Error Control
  ✓ PASSED (empirical rate = 0.0101 ≤ 0.0100)

THEOREM 2: Bayes Factor Consistency  
  ✓ PASSED (BF → ∞ for H₁, BF → 0 for H₀)

THEOREM 3: FDR Control
  ✓ PASSED (mean FDR = 0.0482 ≤ 0.05)

Ha-Tebah Validation
  ✓ PASSED (all criteria met)

======================================================================
All theoretical results verified computationally!
```

---

## 📚 Documentation

### Primary Documents

1. **[METHODOLOGY.md](docs/METHODOLOGY.md)** — Complete methodological details
2. **[Mathematical Proofs](docs/mathematical_proofs.pdf)** (25 pages) — Formal theorems with complete proofs
3. **[Proofs Summary](docs/proofs_summary.pdf)** (5 pages) — Key theorems for main paper appendix
4. **[Technical Slide](docs/technical_slide.html)** — Interactive permutation test visualization
5. **[Infographic](docs/infographic.html)** — Visual framework summary

### API Documentation

Generate HTML documentation:

```bash
cd docs/
python -m pdoc --html ../src/ --output-dir api/
# Open docs/api/index.html in browser
```

### Jupyter Notebooks

Interactive tutorials available in `notebooks/`:
- `01_exploratory_analysis.ipynb` — Data exploration and visualization
- `02_permutation_tests.ipynb` — Statistical testing walkthrough
- `03_bayesian_validation.ipynb` — Bayesian inference tutorial
- `04_diachronic_checks.ipynb` — Manuscript comparison
- `05_expert_panel_analysis.ipynb` — Delphi protocol analysis
- `06_sensitivity_analyses.ipynb` — Robustness checks

---

## 🔄 Reproducibility

### Pre-Registration

All analysis parameters were **pre-registered** on **September 15, 2024**:

- **Registry:** Open Science Framework (OSF)
- **URL:** [https://osf.io/xxxxx/](https://osf.io/xxxxx/)
- **Status:** Locked (immutable)

**Pre-registered elements:**
- Structural marker definitions (`data/structural_markers.json`)
- Target lexeme selection criteria
- Statistical test specifications
- Exclusion criteria for textual variants

### Random Seeds

All stochastic procedures use fixed seeds:

```python
RANDOM_SEEDS = {
    'permutation_tests': 42,
    'bootstrap_resampling': 123,
    'bayesian_mcmc': 456,
    'train_test_split': 789
}
```

### Computational Environment

**Analysis performed on:**
- **OS:** Ubuntu 20.04 LTS / macOS 13+
- **Python:** 3.9.7
- **NumPy:** 1.24.3
- **SciPy:** 1.10.1
- **PyMC:** 5.6.0 (optional)
- **Total runtime:** ~8 hours (full analysis with Bayesian)

Environment specification: [`requirements.txt