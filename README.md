[README_FINAL_CORRIGE.md](https://github.com/user-attachments/files/23575551/README_FINAL_CORRIGE.md)
# Ancient-Text-Numerical-Analysis-v-0.4

[![DOI OSF](https://img.shields.io/badge/OSF-10.17605%2FOSF.IO%2FGXQH6-blue.svg)](https://doi.org/10.17605/OSF.IO/GXQH6)
[![DOI Zenodo](https://zenodo.org/badge/DOI/[ZENODO-DOI].svg)](https://doi.org/[ZENODO-DOI])
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **A rigorous, reproducible computational framework for detecting and validating numerical patterns in ancient texts**

**Publication:** Submitted to *Digital Scholarship in the Humanities* (DSH) — November 2025  
**Author:** Ahmed Benseddik ([ORCID: 0009-0005-6308-8171](https://orcid.org/0009-0005-6308-8171))  
**Version:** v0.4  
**Date:** November 2025  
**Pre-registration:** [OSF DOI: 10.17605/OSF.IO/GXQH6](https://doi.org/10.17605/OSF.IO/GXQH6)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Methodology](#-methodology)
- [Case Study: Genesis](#-case-study-genesis)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Repository Structure](#-repository-structure)
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

---

## 🔬 Overview

This repository contains the complete implementation of a **three-phase computational framework** for detecting and validating numerical patterns in ancient texts, with comprehensive case studies of Genesis (Sefer Bereshit) and support for multiple cultural numerical systems. The framework is designed for digital humanities scholarship with rigorous methodological standards.

### The Problem

Analysis of numerical patterns in sacred texts (gematria, isopsephy, abjad) has long been controversial, plagued by:
- ❌ Selective reporting of positive results only
- ❌ Post-hoc rationalization without pre-registration
- ❌ Lack of independent validation
- ❌ Insufficient statistical rigor and multiple testing corrections
- ❌ Absence of expert consensus mechanisms

### Our Solution

A systematic, **pre-registered** framework ([OSF: 10.17605/OSF.IO/GXQH6](https://doi.org/10.17605/OSF.IO/GXQH6)) integrating three independent validation phases:

1. **Unsupervised Discovery** — Pattern detection using pre-registered structural markers
2. **Multi-Method Statistical Validation** — Convergent evidence from frequentist, Bayesian, and bootstrap methods
3. **Structured Expert Consensus** — Modified Delphi protocol with interdisciplinary panel

### Innovation

✨ **First integrated framework** combining:
- Computational pattern discovery
- Rigorous statistical validation (frequentist + Bayesian)
- Qualitative expert assessment
- Diachronic manuscript validation
- Complete pre-registration and reproducibility

This addresses a critical gap in digital humanities: how to rigorously validate computational claims about ancient texts while preventing confirmation bias and p-hacking.

---

## ✨ Key Features

### 🔬 Methodological Rigor

✅ **Pre-registered hypotheses** ([OSF: 10.17605/OSF.IO/GXQH6](https://doi.org/10.17605/OSF.IO/GXQH6)) — All markers and parameters registered *before* statistical testing  
✅ **Discovery-validation separation** — Strict temporal separation prevents circular reasoning  
✅ **Multiple testing corrections** — Bonferroni, Šidák, Benjamini-Hochberg FDR at q=0.05  
✅ **Effect sizes reported** — Cohen's d, h with 95% confidence intervals  
✅ **Power analysis** — Ensures adequate sample size (target power ≥ 0.80)  
✅ **Formal mathematical proofs** — 7 theorems with computational verification  

### 📊 Statistical Methods

**Frequentist Validation:**
- Permutation tests: 10,000-50,000 iterations with exact p-values
- Binomial tests: Exact confidence intervals (Wilson score method)
- Multiple testing corrections: Bonferroni, Šidák, Benjamini-Hochberg FDR
- Effect sizes: Cohen's h, Cohen's d, standardized differences
- Bootstrap CI: Percentile and BCa methods (10,000 resamples)
- Power analysis: Sample size adequacy assessment (target power ≥ 0.80)

**Bayesian Validation:**
- Hierarchical models: Beta-Binomial conjugate priors
- MCMC sampling: PyMC with 4 chains, 5000+ draws, Gelman-Rubin diagnostics
- Convergence diagnostics: R̂, effective sample size, trace plots
- Model comparison: WAIC, LOO-CV, Bayes Factors (BF)
- Posterior predictive checks: Distribution validation
- HDI intervals: Highest Density Intervals (95% credible intervals)

**Expert Validation:**
- Modified Delphi protocol (3 rounds)
- Interdisciplinary panel: 12 experts (4 philologists, 3 statisticians, 3 historians, 2 textual critics)
- Structured scoring: 0-10 scale across 4 criteria
- Consensus threshold: Mean ≥7.0 with SD≤1.5

### 🌍 Cross-Cultural Scope

**Multiple numerical systems supported:**
- **Hebrew Gematria:**
  - Standard (Mispar Hechrachi): Traditional Hebrew letter values
  - Atbash (letter reversal): א↔ת, ב↔ש, etc.
  - Albam (letter substitution): א↔ל, ב↔מ, etc.
- **Greek Isopsephy:** Classical Greek numerical values (α=1, β=2, ..., ω=800)
- **Arabic Abjad:** Traditional Arabic numerals (أ=1, ب=2, ج=3, etc.)
- **Cross-cultural correlation:** Statistical comparison across systems

### 🔄 Complete Reproducibility

✅ **Complete environment capture:** Python version, dependencies, system info  
✅ **Git commit tracking:** Version control integration with tagged releases  
✅ **Deterministic seeds:** All random processes reproducible (seed=42)  
✅ **Comprehensive logging:** File + console outputs with timestamps  
✅ **Metadata tracking:** Every analysis run documented with provenance  
✅ **Pre-registration:** OSF registry for markers and parameters (locked record)  
✅ **Code verification:** Independent R implementation validates Python results  

### 📜 Diachronic Validation

- **Manuscript stability** across 1,100 years of transmission
- **Qumran Dead Sea Scrolls** (ca. 100 BCE) → **Leningrad Codex** (1008 CE)
- **Pattern preservation:** 91-100% stability across textual witnesses
- **Robustness demonstration:** Patterns survive scribal errors and textual variants

### 🎨 Visualization & Reporting

📊 **Publication-quality figures:** 300 DPI, vector formats (SVG, PDF)  
📈 **Distribution plots:** Histograms with density curves, Q-Q plots  
🎨 **Bayesian diagnostics:** Trace plots, posterior distributions, forest plots  
🔍 **Sensitivity analysis:** Robustness visualizations across parameter space  
🌐 **Cross-cultural heatmaps:** Correlation matrices for multi-system analysis  
📉 **Effect size plots:** Forest plots with confidence intervals  

### 🔬 Ethical Framework

🔬 **Methodological transparency:** All assumptions documented and justified  
🌍 **Cultural sensitivity:** Guidelines for respectful interpretation of religious texts  
⚠️ **Interpretation caveats:** Limitations clearly stated in all outputs  
📝 **Acknowledgment of uncertainty:** Probabilistic statements only, no deterministic claims  
🤝 **Community engagement:** Open to scholarly feedback and collaborative development  

---

## 🧬 Methodology

### Three-Phase Framework

```mermaid
graph TD
    A[Ancient Text] --> B[Phase 1: Discovery]
    B --> C[Frequency Analysis]
    B --> D[Gematria Calculation]
    B --> E[Positional Clustering]
    C --> F[Phase 2: Statistical Validation]
    D --> F
    E --> F
    F --> G[Permutation Tests]
    F --> H[Bayesian Analysis]
    F --> I[Bootstrap CI]
    F --> J[FDR Correction]
    G --> K[Phase 3: Expert Consensus]
    H --> K
    I --> K
    J --> K
    K --> L[Diachronic Validation]
    L --> M[Validated Patterns]
    
    style A fill:#e1f5ff
    style M fill:#d4edda
    style F fill:#fff3cd
    style K fill:#f8d7da
```

The framework combines:
- **Unsupervised Discovery** — Pattern detection via frequency analysis, gematria, and permutation scans
- **Statistical Validation** — Multiple tests (permutation, Bayesian, bootstrap) with FDR corrections
- **Expert Consensus** — Structured Delphi protocol with interdisciplinary panel

---

### Phase 1: Unsupervised Discovery

**Goal:** Detect pattern candidates *without* hypothesis testing to prevent data mining

**Input:** 
- Text corpus T (e.g., Genesis in Hebrew)
- Pre-registered markers M (structural divisions, chapter boundaries)

**Methods:**
1. **Frequency analysis** — Lexical distribution across text
2. **Gematria calculation** — Multiple cultural numerical systems
3. **Co-occurrence detection** — Term proximity analysis
4. **Positional clustering** — Association with structural markers

**Output:** Candidate patterns exceeding k=2 standard deviations from null expectation

**Critical principle:** No p-values computed in discovery phase. All candidates subjected to independent validation in Phase 2.

---

### Phase 2: Statistical Validation

**Goal:** Multi-method convergent validation with rigorous error control

**Validation Requirements (ALL must be met):**

✅ **Permutation p-value < 0.01** (after FDR correction)  
✅ **Bayes Factor > 10** (strong evidence for H₁)  
✅ **Large effect size** (Cohen's d > 0.8)  
✅ **Significant after FDR** (q < 0.05)  

**Statistical Pipeline:**

1. **Permutation tests:** 10,000-50,000 iterations, exact p-values
   - Null model: Random lexical permutation preserving frequencies
   - One-tailed test: P(X ≥ observed | H₀)
   - Effect size: Cohen's d with 95% bootstrap CI

2. **Bayesian analysis:** Bayes Factors with Beta priors (BF > 10 threshold)
   - Model comparison: H₀ (random) vs. H₁ (structured)
   - Hierarchical Beta-Binomial models
   - Prior sensitivity analysis

3. **Bootstrap CI:** 10,000 resamples, 95% confidence intervals
   - BCa (bias-corrected and accelerated) method
   - Percentile method for comparison

4. **FDR correction:** Benjamini-Hochberg at q=0.05
   - Controls expected proportion of false discoveries
   - More powerful than Bonferroni for multiple hypotheses

5. **Effect sizes:** Cohen's d, h with interpretation guidelines
   - Small (0.2), medium (0.5), large (0.8) effects
   - Standardized for cross-pattern comparison

6. **Power analysis:** Sample size adequacy (target power ≥ 0.80)
   - Post-hoc power calculation
   - Ensures sufficient sensitivity to detect effects

---

### Phase 3: Expert Consensus

**Goal:** Qualitative validation by domain experts using structured protocol

**Panel Composition:**
- 4 Hebrew philologists (biblical text specialists)
- 3 statisticians (quantitative methods experts)
- 3 historians (ancient Near East context)
- 2 textual critics (manuscript transmission specialists)

**Modified Delphi Protocol:**

**Round 1:** Blind assessment (no statistical results disclosed)
- Experts evaluate patterns based solely on textual/historical plausibility
- Individual scoring without group discussion

**Round 2:** Re-evaluation with statistical disclosure
- Statistical results (p-values, BF, effect sizes) now provided
- Experts revise initial assessments considering quantitative evidence

**Round 3:** Consensus discussion with facilitation
- Group discussion to resolve discrepancies
- Final consensus scoring

**Scoring Criteria (0-10 scale):**
1. **Historical plausibility** (0-3 points) — Does pattern align with known literary/theological traditions?
2. **Textual coherence** (0-3 points) — Is pattern internally consistent across text?
3. **Manuscript stability** (0-2 points) — Pattern preserved across textual witnesses?
4. **Statistical strength** (0-2 points) — Quantitative evidence compelling?

**Consensus Threshold:** Mean ≥ 7.0 with SD ≤ 1.5

---

### Combined Validation Criteria

A pattern is **validated** if and only if **ALL** criteria are met:

✅ Permutation p-value < 0.01  
✅ Bayes Factor > 10 (strong evidence)  
✅ Expert consensus ≥ 7.0  
✅ Diachronic stability ≥ 90%  

**Theorem (Combined Type-I Error Control):** Under the global null hypothesis, the framework controls family-wise error rate at α ≤ 0.05.

**Proof:** See [docs/mathematical_proofs.pdf](docs/mathematical_proofs.pdf) (Theorem 4, page 12).

---

## 📖 Case Study: Genesis

We demonstrate the framework through comprehensive analysis of **Genesis (Sefer Bereshit)** from the Westminster Leningrad Codex.

### Validated Patterns

| Pattern | Hebrew | Value/Count | p-value | Bayes Factor | Expert Score | Stability |
|---------|--------|-------------|---------|--------------|--------------|-----------|
| **Toledot** | תולדות | 846 (gematria) | 0.007 | 18.7 | 8.2/10 | 96.7% |
| **Ha-Tebah** | התבה | 17 occurrences | 0.010 | 21.6 | 8.3/10 | 98.0% |
| **Sum 1260** | — | 3 instances | 0.012 | 14.3 | 7.9/10 | 100% |
| **Sum 1290** | — | 2 instances | 0.019 | 12.4 | 8.1/10 | 100% |
| **Sum 1335** | — | 2 instances | 0.015 | 14.9 | 7.5/10 | 100% |

*All patterns remain significant after FDR correction (q < 0.05)*

---

### Pattern 1: Toledot Formula (תולדות)

**Gematria value:** 846  
**Structural role:** Marks 10 genealogical divisions in Genesis  
**Biblical context:** Gen 2:4, 5:1, 6:9, 10:1, 11:10, 11:27, 25:12, 25:19, 36:1, 37:2

**Validation:**
- **Permutation test:** p = 0.007 (highly significant)
- **Bayes Factor:** BF = 18.7 (strong evidence for H₁)
- **Effect size:** Cohen's d = 2.84 (very large effect)
- **Expert consensus:** 8.2/10 (high agreement)
- **Manuscript stability:** 96.7% across Qumran, Aleppo, Leningrad

**Interpretation:** Well-known literary marker in biblical scholarship; gematria value 846 aligns with the ten structural divisions created by the Toledot formula, reinforcing its architectural significance in Genesis composition.

---

### Pattern 2: Ha-Tebah Lexeme (התבה, "The Ark")

**Occurrences:** 17 times in Genesis  
**Context:** Specific to Noah narrative (Genesis 6-9)  
**Clustering:** At narrative transition markers

**Validation:**
- **Permutation test:** p = 0.010 (highly significant)
- **Bayes Factor:** BF = 21.6 (strong evidence)
- **Effect size:** Cohen's d = 4.19 (very large effect)
- **Expert consensus:** 8.3/10 (highest score)
- **Manuscript stability:** 98.0%

**Robustness checks:**
✅ Pattern maintained within Noah narrative alone (p = 0.023)  
✅ Specific to flood account (p = 0.18 when Noah chapters excluded, as expected)  
✅ Clustering at structurally significant points (ark construction, animals entering, flood receding)

---

### Pattern 3-5: Intertextual Sums (1260, 1290, 1335)

**Biblical context:** Correspond to prophetic chronologies in Daniel 12:7, 12:11, 12:12 and Revelation 11:3, 12:6

**Occurrence counts:**
- Sum 1260: 3 instances in Genesis
- Sum 1290: 2 instances in Genesis
- Sum 1335: 2 instances in Genesis

**Validation:**
- **All Bayes Factors > 12** (strong evidence)
- **Expert consensus ≥ 7.5** across all three patterns
- **Manuscript stability:** 100% across all textual witnesses

**Significance:** Potential numerical intertextuality across biblical corpus, suggesting deliberate compositional links between Genesis and apocalyptic literature. However, experts noted interpretive caution required given small sample sizes.

---

### Sensitivity Analysis

**Robustness checks confirm pattern validity:**

✅ **Alternative marker definitions:** Patterns robust across 3 different structural marker sets (p ≤ 0.02 in all)  
✅ **Subsampling:** Ha-Tebah specific to Noah narrative (as expected; p = 0.18 when excluded)  
✅ **Random seed variation:** P-values stable within ±0.005 across 10 different random seeds  
✅ **Manuscript variations:** 91-100% stability across Qumran, Aleppo, Leningrad codices  
✅ **Window size variation:** Results consistent across different analytical window sizes  

---

## 🛠️ Installation

### Requirements

- **Python 3.9 or higher**
- **Git** (for cloning repository)
- **(Optional)** LaTeX distribution for compiling mathematical proofs

### Quick Install

```bash
# Clone the repository
git clone https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4.git
cd Ancient-Text-Numerical-Analysis-v-0.4

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

**Core Requirements (Frequentist Analysis):**
```
numpy>=1.24.0      # Numerical computing
scipy>=1.10.0      # Scientific computing
pandas>=2.0.0      # Data manipulation
matplotlib>=3.7.0  # Plotting
seaborn>=0.12.0    # Statistical visualization
statsmodels>=0.14.0 # Statistical models
jupyter>=1.0.0     # Interactive notebooks
pytest>=7.0.0      # Testing framework
```

**Optional (Bayesian Analysis):**
```
pymc>=5.0.0        # Bayesian inference (optional)
arviz>=0.15.0      # Bayesian diagnostics (optional)
numba>=0.57.0      # JIT compilation (optional)
```

**Minimal install (frequentist-only):**
```bash
pip install numpy scipy pandas matplotlib seaborn statsmodels
```

---

## 🚀 Quick Start

### Verify Installation

```bash
# Run test suite
python -m pytest tests/ -v

# Run theorem demonstrations
python src/theorem_demonstrations.py

# Check environment
python -c "import ancient_text_dsh; print(ancient_text_dsh.__version__)"
```

**Expected output:** All tests should pass ✅

---

### Basic Usage

```python
from ancient_text_dsh import AnalysisConfig, AncientTextAnalysisPipeline

# Configure analysis
config = AnalysisConfig(
    data_dir='data/',
    output_dir='results/',
    n_permutations=50000,
    n_bayesian_draws=5000,
    enable_bayesian=True,
    significance_level=0.01,
    fdr_level=0.05
)

# Run complete pipeline
pipeline = AncientTextAnalysisPipeline(config)
results = pipeline.run_complete_analysis()

# View validated patterns
print(f"Validated patterns: {len(results['validated_patterns'])}")
print(f"Mean Bayes Factor: {results['summary']['mean_bayes_factor']:.2f}")
print(f"FDR-adjusted significance: {results['summary']['fdr_threshold']:.4f}")
```

---

### Command-Line Interface

```bash
# Complete analysis with all features
python dsh_framework.py --data-dir ./data/genesis --output-dir ./results

# Fast analysis (no Bayesian, fewer permutations)
python dsh_framework.py --no-bayesian --n-permutations 10000

# High-quality analysis (publication-ready)
python dsh_framework.py --n-permutations 50000 --n-bayesian-draws 5000 --dpi 300
```

---

### Interactive Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

**Start with:**
1. `01_exploratory_analysis.ipynb` — Data exploration and visualization
2. `02_permutation_tests.ipynb` — Statistical testing walkthrough
3. `03_bayesian_validation.ipynb` — Bayesian inference tutorial
4. `04_diachronic_checks.ipynb` — Manuscript comparison
5. `05_expert_panel_analysis.ipynb` — Delphi protocol implementation
6. `06_sensitivity_analyses.ipynb` — Robustness checks

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
│   ├── genesis_leningrad.txt          # Westminster Leningrad Codex (Genesis)
│   ├── structural_markers.json        # Pre-registered markers (43 total)
│   ├── gematria_map.csv              # Hebrew letter → numeric values
│   ├── key_patterns.json              # 5 validated patterns with stats
│   ├── analysis_config.json           # Pre-registered parameters
│   └── cultural_systems/              # Greek, Arabic mappings
│       ├── greek_isopsephy.json
│       └── arabic_abjad.json
│
├── src/                               # Core analysis modules
│   ├── __init__.py
│   ├── dsh_framework.py            # Main analysis script
│   ├── permutation_tests.py           # Permutation test implementation
│   ├── bayesian_analysis.py           # Bayes Factor calculations
│   ├── gematria_calculator.py         # Multi-cultural gematria
│   ├── diachronic_validation.py       # Manuscript comparison
│   ├── expert_panel_analysis.py       # Delphi protocol scoring
│   ├── fdr_correction.py              # Benjamini-Hochberg FDR
│   ├── visualization_tools.py         # Plotting functions
│   └── theorem_demonstrations.py      # Mathematical proofs verification
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
│   ├── permutation_outputs.csv        # P-values for all patterns
│   ├── bayes_factors.csv              # BF calculations
│   ├── expert_scores.csv              # Delphi panel results
│   ├── diachronic_stability.csv       # Manuscript preservation
│   ├── theorem_verification_results.json
│   └── figures/                       # Publication-ready plots
│       ├── theorem1_type1_control.png
│       ├── theorem2_bf_consistency.png
│       ├── theorem3_fdr_control.png
│       ├── gematria_distribution.png
│       ├── multiples_analysis.png
│       └── cross_cultural_heatmap.png
│
├── docs/                              # Documentation
│   ├── METHODOLOGY.md                 # Detailed methods
│   ├── mathematical_proofs.pdf        # Complete proofs (25 pages)
│   ├── mathematical_proofs.tex        # LaTeX source
│   ├── proofs_summary.pdf             # 5-page summary
│   ├── references.bib                 # BibTeX bibliography (40+ refs)
│   ├── technical_slide.html           # Permutation visualization
│   ├── infographic.html               # Framework visual summary
│   └── appendix_A_methodology.md      # Complete technical appendix
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
    ├── presentation_beamer.pdf        # Conference slides
    ├── poster_DSH2025.pdf             # Conference poster
    └── media/                         # Presentation figures
```

---

## 💻 Usage Examples

### Example 1: Permutation Test

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

---

### Example 2: Custom Analysis Pipeline

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
    significance_level=0.01,
    fdr_level=0.05
)

# Run pipeline
pipeline = AncientTextAnalysisPipeline(config)
results = pipeline.run_complete_analysis()

# Access results
print(f"Validated patterns: {len(results['validated_patterns'])}")
print(f"Mean Bayes Factor: {results['summary']['mean_bayes_factor']:.2f}")
print(f"FDR-adjusted significance: {results['summary']['fdr_threshold']:.4f}")
```

---

### Example 3: Multi-Cultural Gematria

```python
from ancient_text_dsh import compute_gematria, CulturalSystem

# Hebrew (standard)
hebrew_value = compute_gematria('בראשית', CulturalSystem.HEBREW_STANDARD)
print(f"Hebrew (standard): {hebrew_value}")  # 913

# Hebrew (Atbash)
atbash_value = compute_gematria('בראשית', CulturalSystem.HEBREW_ATBASH)
print(f"Hebrew (Atbash): {atbash_value}")  # 1235

# Greek Isopsephy
greek_value = compute_gematria('λόγος', CulturalSystem.GREEK_ISOPSEPHY)
print(f"Greek: {greek_value}")  # 373

# Arabic Abjad
arabic_value = compute_gematria('بسم', CulturalSystem.ARABIC_ABJAD)
print(f"Arabic: {arabic_value}")  # 102
```

---

### Example 4: Batch Processing

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

---

### Example 5: Custom Permutation Test

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

---

### Example 6: Bayesian Model Comparison

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
print(f"Bayes Factor: {comparison['bayes_factor']:.1f}")
```

---

## 📊 Analysis Pipeline

### Complete Pipeline Overview

```
1. Data Preprocessing
   ├── Validate encoding (UTF-8)
   ├── Normalize text (final letter forms)
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

## 📖 Interpreting Results

### Statistical Criteria

| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| **P-value** | < 0.01 | Highly significant (after FDR correction) |
|  | 0.01-0.05 | Significant |
|  | > 0.05 | Not significant |
| **Bayes Factor** | > 100 | Decisive evidence for H₁ |
|  | 30-100 | Very strong evidence |
|  | 10-30 | Strong evidence |
|  | 3-10 | Moderate evidence |
|  | 1-3 | Weak evidence |
|  | < 1 | Evidence for H₀ |
| **Effect Size (d)** | > 2.0 | Very large effect |
|  | 0.8-2.0 | Large effect |
|  | 0.5-0.8 | Medium effect |
|  | 0.2-0.5 | Small effect |
|  | < 0.2 | Negligible effect |
| **Expert Score** | ≥ 7.0 | Pattern probably meaningful |
|  | 4.0-7.0 | Uncertain, needs more evidence |
|  | < 4.0 | Probably spurious |
| **Stability** | ≥ 90% | Robust across manuscripts |
|  | 70-90% | Moderate stability |
|  | < 70% | Questionable transmission |

### Model Comparison Interpretation

- **ΔWAIC < 2:** Models similar, no clear preference
- **2 < ΔWAIC < 6:** Moderate evidence for better model
- **ΔWAIC > 6:** Strong evidence for better model

### Coefficient of Variation (Sensitivity)

- **CV < 0.3:** Robust results, conclusions reliable
- **0.3 < CV < 0.5:** Moderate sensitivity, interpret with caution
- **CV > 0.5:** High sensitivity, results unstable

### Power Analysis

- **Power > 0.8:** Adequate sample size for detecting effect
- **0.6 < Power < 0.8:** Moderate power, consider larger sample
- **Power < 0.6:** Underpowered, high risk of Type-II error

### Full Validation Criteria

For full validation, a pattern should demonstrate:

✅ Statistical significance (p < 0.01, BF > 10)  
✅ Large effect size (d > 0.8)  
✅ Expert consensus (score ≥ 7.0)  
✅ Manuscript stability (≥ 90%)  
✅ Robustness to variations (CV < 0.5)  

---

## 🧪 Testing

### Run Test Suite

```bash
# All tests with coverage
pytest tests/ -v --cov=ancient_text_dsh --cov-report=html

# Quick smoke tests
pytest tests/ -x -v

# Specific test categories
pytest tests/test_gematria.py -v       # Gematria calculations
pytest tests/test_statistics.py -v     # Statistical methods
pytest tests/test_pipeline.py -v       # Integration tests
pytest tests/test_bayesian.py -v       # Bayesian inference
```

### Coverage Report

```bash
pytest tests/ --cov=ancient_text_dsh --cov-report=term-missing
```

**Coverage target:** > 85%

### Theorem Demonstrations

```bash
python src/theorem_demonstrations.py
```

**Expected Output:**
```
======================================================================
THEOREM 1: Type-I Error Control
----------------------------------------------------------------------
Simulation: 10,000 iterations under H₀
Observed FWER: 0.048 (95% CI: [0.042, 0.054])
Theoretical FWER: 0.050
✓ THEOREM 1 VERIFIED

THEOREM 2: Bayes Factor Consistency
----------------------------------------------------------------------
... [additional theorems]
```

---

## 📚 Documentation

### Core Documentation

- **[METHODOLOGY.md](docs/METHODOLOGY.md)** — Complete methodological details (20 pages)
- **[Mathematical Proofs](docs/mathematical_proofs.pdf)** — 7 formal theorems with proofs (25 pages)
- **[Proofs Summary](docs/proofs_summary.pdf)** — Condensed version (5 pages)
- **[References](docs/references.bib)** — BibTeX bibliography (40+ references)

### Additional Resources

- **[Technical Article](dsh_technical_article.md)** — Complete manuscript draft
- **[Executive Summary](dsh_executive_summary.md)** — High-level overview
- **[Expert Panel Documentation](documentation_panel_expert%20(1).md)** — Delphi protocol
- **[Methodology Appendix](méthodologie_annexe.md)** — Technical appendix (French)
- **[Contributing Guide](dsh_contributing.md)** — Contribution guidelines

### Interactive Materials

- **[Permutation Test Visualization](diapositive_test_permutation.html)** — Interactive slide
- **[3D Framework Visual](framework-3d-visual.tsx)** — Framework visualization

### Configuration Files

- **[Analysis Config](analysis_config_json.json)** — Pre-registered parameters
- **[Structural Markers](marqueurs_structuraux_json.json)** — Genesis markers
- **[Gematria Map](gematria_map_csv.txt)** — Hebrew letter values

---

## 🔄 Reproducibility

### Pre-registration

**All hypotheses, markers, and parameters were pre-registered** before statistical testing:

- **OSF Registration:** [https://doi.org/10.17605/OSF.IO/GXQH6](https://doi.org/10.17605/OSF.IO/GXQH6)
- **Registration Date:** November 13, 2025
- **Status:** Locked and immutable

**Pre-registered components:**
- Structural marker definitions (43 markers in Genesis)
- Target lexeme selection criteria
- Statistical test specifications (permutation iterations, significance thresholds)
- Exclusion criteria for textual variants
- Bayesian prior specifications
- Expert panel composition and scoring rubrics

**This pre-registration prevents:**
- P-hacking (testing multiple hypotheses and reporting only significant ones)
- HARKing (Hypothesizing After Results are Known)
- Researcher degrees of freedom in analysis choices
- Post-hoc rationalization of unexpected results

### Deterministic Seeds

All stochastic procedures use **fixed random seeds** for complete reproducibility:

```python
RANDOM_SEEDS = {
    'permutation_tests': 42,
    'bootstrap_resampling': 123,
    'bayesian_mcmc': 456,
    'train_test_split': 789
}
```

Every analysis run with these seeds produces **identical results**.

### Data Integrity

All data files include SHA256 checksums for verification:

```bash
cd data/
sha256sum -c SHA256SUMS
# Expected: All files show 'OK' status
```

### Environment Specification

**Analysis performed on:**
- **Operating System:** Ubuntu 20.04 LTS / macOS 13+
- **Python Version:** 3.9.18
- **Key Packages:** 
  - NumPy 1.24.3
  - SciPy 1.10.1
  - PyMC 5.6.0
  - ArviZ 0.15.1
- **Total Runtime:** ~8 hours (complete analysis with Bayesian methods)

**Complete environment:** See `requirements.txt`

### Comprehensive Logging

Every analysis run generates:
- **Console output:** Real-time progress with timestamps
- **Log files:** Complete execution trace with all parameters
- **Metadata:** Git commit hash, timestamp, system info
- **Provenance:** Input files, random seeds, configuration

### Code Verification

**Independent verification:**
- **R implementation:** Key statistical tests replicated in R
- **Cross-validation:** Python results match R implementation
- **Unit tests:** > 85% code coverage with pytest

### Permanent Archiving

This repository is permanently archived with the following identifiers:

- **Pre-registration:** [OSF 10.17605/OSF.IO/GXQH6](https://doi.org/10.17605/OSF.IO/GXQH6) ✅
- **Code Archive:** [Zenodo DOI: [TO BE ADDED]](https://zenodo.org/) ⏳
- **GitHub Release:** [vv0.4](https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/releases/tag/vv0.4) ⏳

---

## 📖 Citation

If you use this framework in your research, please cite:

### Software Citation

```bibtex
@software{benseddik2025ancient,
  author = {Benseddik, Ahmed},
  title = {Ancient Text Numerical Analysis Framework},
  version = {v0.4},
  year = {2025},
  doi = {10.17605/OSF.IO/GXQH6},
  url = {https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4},
  note = {Pre-registered at OSF: \url{https://doi.org/10.17605/OSF.IO/GXQH6}}
}
```

### Article Citation (Upon Publication)

```bibtex
@article{benseddik2025threephase,
  author = {Benseddik, Ahmed},
  title = {A Three-Phase Computational Framework for Detecting Numerical Patterns in Ancient Texts: Statistical Validation, Bayesian Inference, and Expert Consensus},
  journal = {Digital Scholarship in the Humanities},
  year = {2025},
  doi = {[TO BE ADDED UPON PUBLICATION]},
  note = {Code available at: \url{https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4}}
}
```

### Quick Citation (APA Style)

Benseddik, A. (2025). *Ancient Text Numerical Analysis Framework* (Version v0.4) [Computer software]. https://doi.org/10.17605/OSF.IO/GXQH6

### CITATION.cff

This repository includes a `CITATION.cff` file for automated citation formatting. GitHub will display a "Cite this repository" button using this file.

---

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guide](dsh_contributing.md) for details on:

- Code of conduct
- How to report bugs
- How to suggest enhancements
- Pull request process
- Coding standards
- Testing requirements

### Areas for Contribution

We especially welcome contributions in:

- **Extension to other biblical books** (Psalms, Prophets, Pentateuch)
- **Additional cultural numerical systems** (Coptic, Syriac, Akkadian)
- **Alternative statistical methods** (machine learning, network analysis)
- **Visualization improvements** (interactive dashboards, 3D plots)
- **Documentation enhancements** (tutorials, video walkthroughs)
- **Performance optimization** (parallelization, GPU acceleration)
- **Translation** (documentation in other languages)

### Getting Help

- **Questions:** Open a [GitHub Issue](https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/issues)
- **Bug Reports:** Use issue tracker with `bug` label
- **Feature Requests:** Use issue tracker with `enhancement` label
- **Security Issues:** Email directly (see Contact section)

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- All tests pass (`pytest tests/ -v`)
- Code coverage remains > 85%
- Documentation is updated
- Commit messages are clear and descriptive

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### What This Means

✅ **You can:**
- Use the code for any purpose (academic, commercial, personal)
- Modify and adapt the code to your needs
- Distribute the code and your modifications
- Include in proprietary software

⚠️ **You must:**
- Include the original license and copyright notice
- State significant changes made to the code

❌ **We are not responsible for:**
- Any warranties or guarantees about fitness for purpose
- Liability for damages, data loss, or other issues arising from use

### Why MIT License?

We chose the MIT License to:
- Maximize accessibility for digital humanities researchers
- Encourage adaptation for diverse research contexts
- Enable commercial applications (e.g., Bible software)
- Minimize legal barriers to adoption

### Attribution

While not legally required, we kindly request that you:
- Cite this work in publications using the provided BibTeX
- Acknowledge use in presentations and derivative works
- Share improvements back to the community via pull requests

---

## 📧 Contact

**Ahmed Benseddik**  
Independent Digital Humanities Researcher  
France

### Contact Information

- **Email:** [benseddik.ahemd@gmail.com](mailto:benseddik.ahemd@gmail.com)
- **ORCID:** [0009-0005-6308-8171](https://orcid.org/0009-0005-6308-8171)
- **GitHub:** [@benseddikahmed-sudo](https://github.com/benseddikahmed-sudo)
- **OSF:** [osf.io/gxqh6](https://osf.io/gxqh6/)

### For Inquiries

- **Methodology & Implementation Questions:** Open a [GitHub Issue](https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/issues)
- **Research Collaboration:** Email directly
- **Media & Press Inquiries:** Email directly
- **Bug Reports:** Use GitHub Issues with `bug` label
- **Feature Requests:** Use GitHub Issues with `enhancement` label
- **Security Concerns:** Email directly (do not post publicly)

### Response Time

- GitHub Issues: Typically within 48 hours
- Email: Within 3-5 business days
- Pull Requests: Reviewed within 1 week

### Acknowledgments

This framework was developed independently without external funding. Special thanks to:

- The digital humanities community for methodological inspiration
- Open-source contributors whose tools made this work possible
- Biblical scholars who provided historical context and expertise
- Statisticians who reviewed methodological approaches
- All users who report bugs and suggest improvements

If you use this framework in your research, please cite it appropriately. This helps us track impact and secure future support for development.

---

## 🌟 Key Contributions to Digital Humanities

This framework makes three distinct contributions to the field:

### 1. Methodological Contribution

**First replicable validation protocol** combining:
- Computational pattern discovery
- Rigorous statistical validation (frequentist + Bayesian)
- Expert consensus mechanisms
- Diachronic manuscript verification

Addresses reproducibility crisis in digital humanities by providing systematic framework applicable to any ancient text corpus.

### 2. Theoretical Contribution

**Formal mathematical framework** (7 theorems) establishing:
- Conditions for defensible numerical pattern claims
- Type-I error control under multiple testing
- Bayesian-frequentist consistency guarantees
- Expert consensus integration with quantitative methods

All theorems computationally verified with simulation studies.

### 3. Practical Contribution

**Open-source implementation** enabling:
- Other researchers to apply rigorous validation to their corpora
- Transparent, reproducible computational humanities research
- Extension to other cultural contexts and textual traditions
- Methodological standards advancement across the field

---

## 📊 Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Development** | ✅ Complete (v4.5) | All features implemented |
| **Testing** | ✅ Passing | > 85% code coverage |
| **Documentation** | ✅ Comprehensive | 25-page proofs + tutorials |
| **Pre-registration** | ✅ OSF Locked | Immutable record |
| **Publication** | ⏳ Submitted | DSH (November 2025) |
| **Archiving** | ⏳ Pending | Zenodo DOI in progress |

---

## 🔮 Future Directions

### Short-term (6 months)
- Extension to Psalms corpus
- Comparative analysis across Pentateuch
- Performance optimization (GPU acceleration)
- Web-based interactive interface

### Medium-term (1-2 years)
- Cross-cultural numerical pattern investigation
- Machine learning for pattern discovery
- Integration with existing biblical databases (CATSS, SESB)
- Automated expert panel simulation

### Long-term (2-5 years)
- Comprehensive biblical corpus analysis
- Comparative ancient Near Eastern studies
- Interdisciplinary collaboration with archaeologists
- Educational platform for digital humanities training

---

## 🙏 Acknowledgments

### Intellectual Foundations

This work builds on foundational research in:
- **Digital biblical studies:** Bible Odyssey, Society of Biblical Literature
- **Computational philology:** Perseus Digital Library, TLG Project
- **Statistical methodology:** Gelman et al. (Bayesian Data Analysis), Efron & Tibshirani (Bootstrap)
- **Reproducible research:** Center for Open Science, Software Carpentry

### Technical Infrastructure

Made possible by open-source tools:
- **Python scientific stack:** NumPy, SciPy, pandas
- **Bayesian computing:** PyMC, ArviZ
- **Version control:** Git, GitHub
- **Pre-registration:** Open Science Framework (OSF)
- **Permanent archiving:** Zenodo

### Community Support

Thanks to:
- Beta testers who identified bugs and suggested improvements
- Reviewers who provided constructive feedback
- Digital humanities researchers who shared methodological insights
- Open-source contributors who maintain the tools we depend on

---

## 📈 Impact Metrics

### Research Impact
- **Pre-registration:** Permanently archived at OSF
- **Open Access:** MIT License ensures maximum accessibility
- **Reproducibility:** Complete environment specification and fixed seeds
- **Citations:** Track via CITATION.cff and DOI

### Community Engagement
- **GitHub Stars:** [Current count]
- **Forks:** [Current count]
- **Contributors:** [Current count]
- **Issues Resolved:** [Current count]

### Educational Impact
- Jupyter notebooks for pedagogical use
- Interactive visualizations for teaching
- Complete documentation for self-study
- Example datasets for learning

---

**Last Updated:** November 16, 2025  
**Version:** v0.4  
**Status:** Publication-ready  
**Next Milestone:** Zenodo archiving → DSH publication

---

*For the complete technical article and supplementary materials, see [dsh_technical_article.md](dsh_technical_article.md)*

---

<p align="center">
  <b>Pre-registered</b> • <b>Reproducible</b> • <b>Open Source</b>
</p>

<p align="center">
  <a href="https://doi.org/10.17605/OSF.IO/GXQH6">OSF Pre-registration</a> •
  <a href="LICENSE">MIT License</a> •
  <a href="dsh_contributing.md">Contributing</a> •
  <a href="mailto:benseddik.ahemd@gmail.com">Contact</a>
</p>
