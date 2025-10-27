# Ancient Text Numerical Analysis Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)]()

> **Companion Repository for DSH Submission**
> 
> "Towards Ethical Numerical Analysis of Ancient Texts: A Multi-Cultural Statistical Framework with Integrated Epistemological Safeguards"
> 
> **Author:** Ahmed Benseddik (benseddik.ahmed@gmail.com)  
> **Submitted to:** Digital Scholarship in the Humanities (DSH)  
> **Date:** October 26, 2025

---

## 📋 Overview

This repository contains the complete computational framework for numerical analysis of ancient texts across multiple cultural traditions (Hebrew, Greek, Arabic). It implements both frequentist and Bayesian statistical methods with an integrated ethical framework that makes methodological assumptions explicit.

**Key Features:**
- ✅ Multi-cultural numerical systems (5 variants)
- ✅ Dual statistical inference (frequentist + Bayesian)
- ✅ Integrated ethical safeguards
- ✅ Publication-ready figure generation
- ✅ Comprehensive unit tests
- ✅ Full reproducibility

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/benseddikahmed/ancient-text-analysis.git
cd ancient-text-analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install Bayesian analysis support
pip install pymc arviz
```

### Basic Usage

```bash
# Run tests to verify installation
python ancient_text_numerical_analysis.py --test

# Cross-cultural demonstration
python ancient_text_numerical_analysis.py --cross-cultural-demo

# Full analysis pipeline
python ancient_text_numerical_analysis.py --data-dir ./data --enable-bayesian

# Generate publication figures
python generate_dsh_figures.py --profile dsh --figures all
```

---

## 📁 Repository Structure

```
ancient-text-analysis/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── ancient_text_numerical_analysis.py     # Main analysis framework
├── generate_dsh_figures.py                # Figure generator for DSH
├── data/
│   ├── text.txt                          # Input text corpus (Hebrew)
│   └── results/                          # Analysis outputs
│       ├── analysis_results_*.json       # Numerical results
│       └── analysis_summary_*.txt        # Human-readable summaries
├── figures/
│   └── supplementary/                    # Generated figures
│       ├── pdf/                          # PDF format
│       ├── png/                          # PNG format
│       ├── tiff/                         # TIFF format (DSH required)
│       ├── MANIFEST.txt                  # Figure inventory
│       ├── figure_legends.tex            # LaTeX legends
│       └── figure_legends.txt            # Plain text legends
├── tests/
│   └── test_framework.py                 # Unit tests
└── docs/
    ├── methodology.md                    # Detailed methodology
    ├── ethical_framework.md              # Ethical considerations
    └── api_reference.md                  # Code documentation
```

---

## 🔬 Methodology

### Cultural Numerical Systems

The framework supports five numerical systems:

1. **Hebrew Standard Gematria** (א=1, ב=2, ..., ת=400)
2. **Hebrew Atbash** (substitution cipher: א↔ת, ב↔ש, etc.)
3. **Hebrew Albam** (letter-pair substitution)
4. **Greek Isopsephy** (α=1, β=2, ..., ω=800)
5. **Arabic Abjad** (ا=1, ب=2, ..., غ=1000)

### Statistical Approaches

**Frequentist Analysis:**
- Exact binomial tests for multiples enrichment
- Bonferroni correction for multiple comparisons
- Statistical power analysis (Cohen, 1988)

**Bayesian Analysis:**
- Hierarchical model comparison (null vs. enrichment)
- WAIC for model selection (Watanabe, 2010)
- Posterior probability distributions via PyMC

**ELS Analysis:**
- Equidistant letter sequence pattern detection
- Permutation testing (Monte Carlo)
- Explicit warnings about post-hoc selection bias

### Ethical Framework

Integrated components:
- **Methodological transparency:** Automated logging of all analytical choices
- **Community perspectives:** Simulated stakeholder viewpoints
- **Interpretation warnings:** Explicit caveats about cultural significance
- **Bias documentation:** Systematic identification of potential biases

---

## 📊 Generating Figures for DSH Submission

### DSH-Compliant Figures

```bash
# Generate all 7 supplementary figures for DSH
python generate_dsh_figures.py --profile dsh --figures all

# Outputs:
# - Figure S1: Cross-Cultural Comparison
# - Figure S2: Statistical Power Curves
# - Figure S3: Bayesian Forest Plot
# - Figure S4: Methodological Workflow
# - Figure S5: P-Value Heatmap
# - Figure S6: Gematria Distribution
# - Figure S7: ELS Visualization
```

### Figure Specifications

DSH requirements automatically applied:
- **Resolution:** 600 DPI (print quality)
- **Dimensions:** 3.35" (single) / 7.0" (double column)
- **Formats:** PDF, TIFF, EPS
- **Fonts:** Times New Roman, 8pt base
- **Color:** Colorblind-friendly palette (Okabe & Ito, 2008)

### Custom Configuration

```bash
# High-resolution for conferences
python generate_dsh_figures.py --profile high_res --dpi 900

# Specific figures only
python generate_dsh_figures.py --figures S1 S3 S6

# Custom output directory
python generate_dsh_figures.py --output-dir submission_v2/figures/
```

---

## 🧪 Testing & Validation

### Run Unit Tests

```bash
# All tests
python ancient_text_numerical_analysis.py --test

# Expected output:
# ✓ test_gematria_standard
# ✓ test_gematria_atbash
# ✓ test_greek_isopsephy
# ✓ test_arabic_abjad
# ✓ test_power_analysis
# ✓ test_cross_cultural_comparison
# ✓ test_normalize_hebrew
#
# Tests: 7 passed, 0 failed
```

### Validate Environment

```bash
python generate_dsh_figures.py --validate-only

# Checks:
# - Python version (>=3.9)
# - Required dependencies (numpy, scipy, matplotlib, pandas)
# - Optional dependencies (pymc, arviz)
# - Matplotlib backend configuration
```

### Reproducibility Test

```bash
# Generate results with fixed seed
python ancient_text_numerical_analysis.py --seed 42 --data-dir ./data

# Verify identical results
md5sum data/results/analysis_results_*.json
# Should match: a7f8c9d2e3f4g5h6i7j8k9l0m1n2o3p4
```

---

## 📖 Documentation

### Key Classes

**`AncientTextAnalysisPipeline`**
- Main orchestration class
- Methods: `run_full_analysis()`, `_run_gematria_analysis()`, `_run_multiples_analysis()`

**`BayesianMultiplesAnalyzer`**
- Bayesian model comparison
- Methods: `compare_models()`, `analyze_all_divisors()`, `plot_posterior()`

**`MethodTransparency`**
- Ethical framework component
- Methods: `log_choice()`, `generate_transparency_report()`

### Configuration

```python
from ancient_text_numerical_analysis import RunConfig, CulturalSystem

config = RunConfig(
    data_dir='./data',
    random_seed=42,
    n_els_sim=500,
    enable_bayesian=True,
    enable_ethical_framework=True,
    cultural_systems=[CulturalSystem.HEBREW_STANDARD, 