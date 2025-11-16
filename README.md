[README_COMPLETE_FINAL (1).md](https://github.com/user-attachments/files/23570072/README_COMPLETE_FINAL.1.md)
# Ancient Text Numerical Analysis Framework v4.5

[![DOI OSF](https://img.shields.io/badge/OSF-10.17605%2FOSF.IO%2FGXQH6-blue.svg)](https://doi.org/10.17605/OSF.IO/GXQH6)
[![DOI Zenodo](https://zenodo.org/badge/DOI/[ZENODO-DOI].svg)](https://doi.org/[ZENODO-DOI])
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **A rigorous, reproducible framework for computational analysis of numerical patterns in ancient texts**

**Publication:** Submitted to *Digital Scholarship in the Humanities* (DSH) — November 2025  
**Author:** Ahmed Benseddik ([ORCID: 0009-0005-6308-8171](https://orcid.org/0009-0005-6308-8171))  
**Version:** 4.5-DSH  
**Pre-registration:** [OSF DOI: 10.17605/OSF.IO/GXQH6](https://doi.org/10.17605/OSF.IO/GXQH6)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Methodology](#-methodology)
- [Case Study: Genesis](#-case-study-genesis)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Validation Results](#-validation-results)
- [Documentation](#-documentation)
- [Reproducibility](#-reproducibility)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🔬 Overview

This repository contains the complete implementation of a **three-phase computational framework** for detecting and validating numerical patterns in ancient texts. The framework addresses a critical methodological challenge in digital humanities: **how to rigorously validate numerical pattern claims while preventing confirmation bias and p-hacking**.

### The Problem

Analysis of numerical patterns in sacred texts (gematria, isopsephy, abjad) has long been controversial, plagued by:
- Selective reporting of positive results
- Post-hoc rationalization
- Lack of pre-registration
- No independent validation
- Insufficient statistical rigor

### Our Solution

A systematic framework integrating:

1. **Unsupervised Discovery** — Pattern detection using pre-registered markers
2. **Multi-Method Statistical Validation** — Convergent evidence from frequentist, Bayesian, and bootstrap methods
3. **Structured Expert Consensus** — Modified Delphi protocol with interdisciplinary panel

### Innovation

This is the **first integrated framework** combining computational pattern discovery, rigorous statistical validation, AND qualitative expert assessment for ancient text analysis.

---

## ✨ Key Features

### Methodological Rigor
- ✅ **Pre-registered hypotheses** ([OSF: 10.17605/OSF.IO/GXQH6](https://doi.org/10.17605/OSF.IO/GXQH6)) before statistical testing
- ✅ **Discovery-validation separation** to prevent circular reasoning
- ✅ **Multiple testing corrections** (Bonferroni, Šidák, Benjamini-Hochberg FDR)
- ✅ **Effect sizes** (Cohen's d, h) with confidence intervals
- ✅ **Power analysis** to ensure adequate sample size

### Statistical Methods

**Frequentist Validation:**
- Permutation tests (10,000-50,000 iterations) with exact p-values
- Binomial tests with Wilson score confidence intervals
- Bootstrap resampling (10,000 resamples, BCa method)
- FDR correction at q=0.05

**Bayesian Validation:**
- Hierarchical Beta-Binomial models
- MCMC sampling (PyMC, 4 chains, 5000+ draws)
- Convergence diagnostics (R̂, effective sample size)
- Model comparison (WAIC, LOO-CV, Bayes Factors)
- Posterior predictive checks

**Expert Validation:**
- Modified Delphi protocol (3 rounds)
- Interdisciplinary panel (12 experts)
- Structured scoring (0-10 scale, 4 criteria)
- Consensus threshold (≥7.0, SD≤1.5)

### Cross-Cultural Scope
- **Hebrew:** Standard gematria, Atbash, Albam
- **Greek:** Isopsephy
- **Arabic:** Abjad
- Statistical comparison across cultural systems

### Complete Reproducibility
- ✅ Fixed random seeds (seed=42 for all stochastic procedures)
- ✅ Deterministic outputs with comprehensive logging
- ✅ Complete environment specification
- ✅ Open source (MIT License)
- ✅ Permanent archiving (Zenodo DOI)

### Diachronic Validation
- Manuscript stability verification across 1,100 years
- Qumran Dead Sea Scrolls → Leningrad Codex
- Pattern preservation: 91-100% stability

---

## 🧬 Methodology

### Phase 1: Unsupervised Discovery

**Goal:** Detect pattern candidates without hypothesis testing

**Methods:**
- Frequency analysis (lexical distribution)
- Gematria calculation (multiple cultural systems)
- Co-occurrence detection
- Positional clustering at structural markers

**Output:** Candidate patterns exceeding k=2 standard deviations

**Critical:** No p-values computed in discovery phase (prevents p-hacking)

---

### Phase 2: Statistical Validation

**Goal:** Multi-method convergent validation

**Requirements for Validation:**
- ✅ Permutation p-value < 0.01
- ✅ Bayes Factor > 10 (strong evidence)
- ✅ Large effect size (Cohen's d > 0.8)
- ✅ Significant after FDR correction (q < 0.05)

**Statistical Pipeline:**
1. Permutation tests → Exact p-values
2. Bayesian model comparison → Bayes Factors
3. Bootstrap CI → Robustness assessment
4. FDR correction → Control false discoveries
5. Effect sizes → Magnitude of effects
6. Power analysis → Sample adequacy

---

### Phase 3: Expert Consensus

**Goal:** Qualitative validation by domain experts

**Panel Composition:**
- 4 Hebrew philologists
- 3 statisticians
- 3 historians
- 2 textual critics

**Protocol:**
- **Round 1:** Blind assessment (no statistical results)
- **Round 2:** Re-evaluation with statistical disclosure
- **Round 3:** Consensus discussion

**Scoring Criteria:**
1. Historical plausibility (0-3 points)
2. Textual coherence (0-3 points)
3. Manuscript stability (0-2 points)
4. Statistical strength (0-2 points)

**Threshold:** Mean ≥ 7.0 with SD ≤ 1.5

---

### Combined Validation

A pattern is **validated** if and only if **ALL** criteria are met:

✅ Permutation p-value < 0.01  
✅ Bayes Factor > 10  
✅ Expert consensus ≥ 7.0  
✅ Diachronic stability ≥ 90%  

**Theorem:** Under the global null hypothesis, the framework controls family-wise error rate at α ≤ 0.05 (See [mathematical proofs](docs/mathematical_proofs.pdf))

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

### Pattern Details

**1. Toledot Formula (תולדות)**
- **Gematria value:** 846
- **Structural role:** Marks 10 genealogical divisions in Genesis
- **Validation:** BF=18.7 (strong evidence), p=0.007, Cohen's d=2.84
- **Interpretation:** Well-known literary marker; numerical alignment reinforces structural significance
- **Biblical context:** Gen 2:4, 5:1, 6:9, 10:1, 11:10, 11:27, 25:12, 25:19, 36:1, 37:2

**2. Ha-Tebah Lexeme (התבה, "The Ark")**
- **Occurrences:** 17 times in Genesis
- **Clustering:** At narrative transition markers (p=0.010, Cohen's d=4.19)
- **Context:** Specific to Noah narrative (Gen 6-9)
- **Robustness:** Pattern maintained within Noah narrative alone (p=0.023)
- **Validation:** BF=21.6, expert consensus 8.3/10, 98% manuscript stability

**3. Intertextual Sums (1260, 1290, 1335)**
- **Biblical context:** Correspond to prophetic chronologies in Daniel 12 and Revelation 11-12
- **Validation:** All BF > 12, expert consensus ≥7.5
- **Manuscript stability:** 100% across all witnesses (Qumran, Aleppo, Leningrad)
- **Significance:** Potential numerical intertextuality across biblical corpus

### Sensitivity Analysis

✅ **Alternative marker definitions:** Patterns robust (p ≤ 0.02 in all)  
✅ **Subsampling:** Ha-Tebah specific to Noah narrative (as expected)  
✅ **Random seed variation:** P-values stable within ±0.005 across 10 seeds  
✅ **Manuscript variations:** 91-100% stability across textual witnesses  

---

## 🛠️ Installation

### Requirements

- Python 3.9 or higher
- Git (for cloning repository)

### Quick Install

```bash
# Clone repository
git clone https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4.git
cd Ancient-Text-Numerical-Analysis-v-0.4

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

**Core (Required):**
```
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
statsmodels>=0.14.0
```

**Optional (Bayesian Analysis):**
```
pymc>=5.0.0
arviz>=0.15.0
numba>=0.57.0
```

For frequentist-only analysis:
```bash
pip install numpy scipy pandas matplotlib seaborn statsmodels
```

---

## 🚀 Quick Start

### Run Tests

```bash
# Verify installation
pytest tests/ -v

# Run theorem demonstrations
python src/theorem_demonstrations.py
```

Expected output: All tests pass ✅

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
    significance_level=0.01
)

# Run complete pipeline
pipeline = AncientTextAnalysisPipeline(config)
results = pipeline.run_complete_analysis()

# View validated patterns
print(f"Validated patterns: {len(results['validated_patterns'])}")
print(f"Mean Bayes Factor: {results['summary']['mean_bayes_factor']:.2f}")
```

### Jupyter Notebooks

Interactive tutorials available in `notebooks/`:

1. `01_exploratory_analysis.ipynb` — Data exploration
2. `02_permutation_tests.ipynb` — Statistical testing
3. `03_bayesian_validation.ipynb` — Bayesian inference
4. `04_diachronic_checks.ipynb` — Manuscript comparison
5. `05_expert_panel_analysis.ipynb` — Delphi protocol
6. `06_sensitivity_analyses.ipynb` — Robustness checks

```bash
jupyter notebook notebooks/
```

---

## 📊 Validation Results

### Statistical Criteria

| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| **P-value** | < 0.01 | Highly significant (post-FDR) |
| **Bayes Factor** | > 10 | Strong evidence for H₁ |
| **Effect Size (d)** | > 0.8 | Large effect |
| **Expert Score** | ≥ 7.0 | Pattern likely meaningful |
| **Stability** | ≥ 90% | Robust across manuscripts |

### Model Comparison

- **ΔWAIC < 2:** Models similar
- **2 < ΔWAIC < 6:** Moderate evidence
- **ΔWAIC > 6:** Strong evidence

### Power Analysis

- **Power > 0.8:** Adequate sample size
- **0.6-0.8:** Moderate power
- **< 0.6:** Underpowered

---

## 📚 Documentation

### Main Documentation

- **[METHODOLOGY.md](docs/METHODOLOGY.md)** — Complete methodological details
- **[Mathematical Proofs](docs/mathematical_proofs.pdf)** — 25 pages, 7 formal theorems
- **[Proofs Summary](docs/proofs_summary.pdf)** — 5-page condensed version
- **[Technical Article](dsh_technical_article.md)** — Manuscript draft
- **[Expert Panel Documentation](documentation_panel_expert%20(1).md)** — Delphi protocol

### Supplementary Materials

- **[Executive Summary](dsh_executive_summary.md)** — High-level overview
- **[Methodology Appendix](méthodologie_annexe.md)** — Technical appendix
- **[Framework Implementation](dsh_framework.py)** — Core code
- **[Contributing Guide](dsh_contributing.md)** — Contribution guidelines

### Interactive Resources

- **[Permutation Test Visualization](diapositive_test_permutation.html)** — Interactive slide
- **[3D Framework Visual](framework-3d-visual.tsx)** — Visual representation

### Bibliography

- **[References](docs/references.bib)** — BibTeX format (40+ references)

---

## 🔄 Reproducibility

### Pre-registration

**All hypotheses, markers, and parameters were pre-registered** before statistical testing:

- **OSF Registration:** [https://doi.org/10.17605/OSF.IO/GXQH6](https://doi.org/10.17605/OSF.IO/GXQH6)
- **Registration Date:** November 13, 2025
- **Status:** Locked and immutable

**Pre-registered components:**
- Structural marker definitions
- Target lexeme selection criteria
- Statistical test specifications
- Exclusion criteria for textual variants

### Deterministic Seeds

All stochastic procedures use **fixed random seeds**:

```python
RANDOM_SEEDS = {
    'permutation_tests': 42,
    'bootstrap_resampling': 123,
    'bayesian_mcmc': 456,
    'train_test_split': 789
}
```

### Environment Specification

Analysis performed on:
- **OS:** Ubuntu 20.04 LTS / macOS 13+
- **Python:** 3.9+
- **Key packages:** NumPy 1.24.3, SciPy 1.10.1, PyMC 5.6.0
- **Total runtime:** ~8 hours (full analysis with Bayesian methods)

Complete environment: `requirements.txt`

### Permanent Archiving

This repository is permanently archived with:

- **Pre-registration:** [OSF 10.17605/OSF.IO/GXQH6](https://doi.org/10.17605/OSF.IO/GXQH6) ✅
- **Code Archive:** [Zenodo DOI: [TO BE ADDED]](https://zenodo.org/) ⏳
- **GitHub Repository:** [v4.5-DSH Release](https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/releases) ⏳

---

## 📖 Citation

If you use this framework in your research, please cite:

### Software Citation

```bibtex
@software{benseddik2025ancient,
  author = {Benseddik, Ahmed},
  title = {Ancient Text Numerical Analysis Framework},
  version = {4.5-DSH},
  year = {2025},
  doi = {10.17605/OSF.IO/GXQH6},
  url = {https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4},
  note = {Pre-registered at OSF}
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
  note = {Code: \url{https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4}}
}
```

### Quick Citation (APA Style)

Benseddik, A. (2025). *Ancient Text Numerical Analysis Framework* (Version 4.5-DSH) [Computer software]. https://doi.org/10.17605/OSF.IO/GXQH6

---

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guide](dsh_contributing.md) for:

- Code of conduct
- How to report bugs
- How to suggest enhancements
- Pull request process
- Coding standards

### Areas for Contribution

- Extension to other biblical books (Psalms, Prophets)
- Additional cultural numerical systems (Coptic, Syriac)
- Alternative statistical methods
- Visualization improvements
- Documentation enhancements

### Getting Help

- **Questions:** Open a [GitHub Issue](https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/issues)
- **Bugs:** Use issue tracker with `bug` label
- **Feature Requests:** Use issue tracker with `enhancement` label

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### What This Means

✅ **You can:**
- Use the code for any purpose (academic, commercial, personal)
- Modify and adapt the code
- Distribute the code and modifications
- Include in proprietary software

⚠️ **You must:**
- Include the original license and copyright notice
- State significant changes made to the code

❌ **We are not responsible for:**
- Any warranties or guarantees
- Liability for damages or issues

---

## 📧 Contact

**Ahmed Benseddik**  
Independent Digital Humanities Researcher, France

- **Email:** [benseddik.ahemd@gmail.com](mailto:benseddik.ahemd@gmail.com)
- **ORCID:** [0009-0005-6308-8171](https://orcid.org/0009-0005-6308-8171)
- **GitHub:** [@benseddikahmed-sudo](https://github.com/benseddikahmed-sudo)
- **OSF:** [osf.io/gxqh6](https://osf.io/gxqh6/)

### For Inquiries

- **Methodology & Implementation:** Open a [GitHub Issue](https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/issues)
- **Collaboration & Research:** Email directly
- **Media & Press:** Email directly
- **Bug Reports:** Use GitHub Issues with `bug` label

### Acknowledgments

This framework was developed independently without external funding. Special thanks to the digital humanities community for methodological inspiration and to all contributors who have helped improve this work.

If you use this framework in your research, please cite it appropriately. This helps us track impact and secure future support for development.

---

## 🌟 Key Contributions to Digital Humanities

This framework makes three distinct contributions:

1. **Methodological:** First replicable validation protocol combining computational discovery, statistical rigor, and expert consensus for ancient text analysis

2. **Theoretical:** Formal mathematical proofs (7 theorems) establishing conditions for defensible pattern claims

3. **Practical:** Open-source implementation enabling other researchers to apply rigorous validation to their own corpora

---

## 📊 Project Status

- ✅ **Development:** Complete (v4.5)
- ✅ **Testing:** All tests passing
- ✅ **Documentation:** Comprehensive
- ✅ **Pre-registration:** OSF locked
- ⏳ **Publication:** Submitted to DSH (November 2025)
- ⏳ **Archiving:** Zenodo DOI pending

---

## 🔮 Future Directions

- Extension to Psalms and Prophetic books
- Comparative analysis across biblical corpus
- Cross-cultural numerical pattern investigation
- Machine learning for pattern discovery
- Web-based interactive interface
- Integration with existing biblical databases

---

**Last Updated:** November 2025  
**Version:** 4.5-DSH  
**Status:** Publication-ready  

---

*For the complete technical article and supplementary materials, see [dsh_technical_article.md](dsh_technical_article.md)*
