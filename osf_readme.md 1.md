# A Three-Phase Statistical Framework for Computational Analysis of Numerical Patterns in Ancient Texts

[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FGXQH6-blue)](https://doi.org/10.17605/OSF.IO/GXQH6)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Authors:** BENSEDDIK. AHMED 
**Institution:** [Institution Name]  
**Contact:** benseddik.ahmed@gmail.com  
**Date:** December 2025  
**Version:** 1.0.0

---

## Overview

This repository contains complete materials for a comprehensive statistical framework addressing methodological challenges in computational analysis of numerical patterns in ancient texts. The framework integrates exploratory pattern discovery, dual-paradigm statistical validation (frequentist and Bayesian), and structured expert hermeneutical assessment.

**Key Contribution:** We demonstrate that rigorous methods, when applied to Hebrew Genesis, yield well-justified null results—advancing knowledge through principled rejection of weak evidence rather than inflated positive claims.

**Critical Case Study:** Analysis of תולדות ('generations') reveals how preprocessing normalization creates spurious patterns by obscuring manuscript orthographic variation (gematria values 846, 840, 834), demonstrating that statistical validation requires philological expertise.

---

## Repository Contents

### 📄 Main Manuscript
- `manuscript_main.pdf` - Complete manuscript (11,500 words, 45-50 pages)
- `manuscript_main.docx` - Editable Word version
- `manuscript_main.tex` - LaTeX source (if applicable)

### 📋 Supplementary Materials
- `supplementary_materials.pdf` - Complete supplementary documentation (25,000 words, 60-70 pages)
- `supplementary_materials.md` - Markdown version for easy viewing
- Sections include:
  - S1: Technical Implementation Guide
  - S2: Extended Biblical Analysis
  - S3: Validation Study Details
  - S4: Ethical Framework Documentation
  - S5: Expert Evaluation Materials
  - S6: Additional Figures and Visualizations
  - S7: Reproducibility Verification
  - S8: Glossary of Technical Terms
  - S9: Frequently Asked Questions
  - S10: Additional Resources

### 💻 Code
- `code/` - Complete Python implementation
  - `gematria_framework/` - Main package
    - `phase1_discovery.py`
    - `phase2_validation.py`
    - `phase3_expertise.py`
    - `gematria.py` - Hebrew, Greek, Arabic systems
    - `preprocessing.py`
    - `statistics.py`
    - `visualization.py`
    - `utils.py`
  - `tests/` - Comprehensive test suite (85% coverage)
  - `requirements.txt` - Pinned dependencies
  - `setup.py` - Package installation
  - `Dockerfile` - Container specification

### 📊 Data
- `data/` - Input texts and analysis results
  - `genesis_wlc.txt` - Westminster Leningrad Codex (Genesis)
  - `genesis_wlc_sha256.txt` - Checksum for verification
  - `genesis_preprocessed.json` - Processed segments
  - `genesis_gematria_values.csv` - Computed values
  - `toledot_manuscript_comparison.csv` - Orthographic variation data

### 📈 Results
- `results/` - Complete analysis outputs
  - `phase1_exploration/` - Descriptive statistics, visualizations
  - `phase2_validation/` - Statistical test results
    - `frequentist_results.csv`
    - `bayesian_posteriors.csv`
    - `sensitivity_analysis.csv`
    - `mcmc_traces.pkl`
  - `figures/` - All generated plots (PNG, PDF)
  - `tables/` - Publication-ready tables (CSV, LaTeX)

### 📓 Tutorials
- `tutorials/` - Interactive Jupyter notebooks
  - `01_quick_start.ipynb` - 15-minute introduction
  - `02_phase1_discovery.ipynb` - Exploratory analysis
  - `03_phase2_validation.ipynb` - Statistical testing
  - `04_manuscript_verification.ipynb` - Orthographic analysis
  - `05_sensitivity_analysis.ipynb` - Robustness testing
  - `06_custom_systems.ipynb` - Greek, Arabic examples

### 📹 Video Documentation
- `videos/` - Tutorial recordings
  - `installation_setup.mp4` (15 min)
  - `quick_start_tutorial.mp4` (30 min)
  - `complete_workflow.mp4` (90 min)

### 📚 Documentation
- `docs/` - Full API reference and guides
  - `API_reference.md`
  - `installation_guide.md`
  - `user_manual.md`
  - `developer_guide.md`
  - `ethical_guidelines.md`

### 🔧 Reproducibility Infrastructure
- `environment.yml` - Conda environment specification
- `Dockerfile` - Container for exact replication
- `docker-compose.yml` - Multi-container orchestration
- `.github/workflows/` - CI/CD pipelines
- `CITATION.cff` - Citation metadata

---

## Quick Start

### Installation (5 minutes)

**Option 1: pip (recommended for users)**
```bash
pip install gematria-framework
```

**Option 2: Docker (recommended for reproducibility)**
```bash
docker pull ghcr.io/username/gematria-framework:v1.0
docker run -v $(pwd)/data:/data -p 8888:8888 gematria-framework
# Access Jupyter at http://localhost:8888
```

**Option 3: From source (recommended for developers)**
```bash
git clone https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4.git
cd gematria-framework
pip install -e ".[dev]"
pytest tests/  # Verify installation
```

### Basic Usage (2 minutes)

```python
from gematria_framework import HebrewText, Phase1Discovery, Phase2Validation

# Load text
text = HebrewText.from_file("data/genesis_wlc.txt")

# Phase 1: Exploratory Discovery
discovery = Phase1Discovery(text)
discovery.set_window_parameters(window_size=5, stride=5)
discovery.compute_gematria()
discovery.plot_distribution(save_path="figures/distribution.png")

# Phase 2: Statistical Validation
candidates = discovery.identify_candidates(divisors=[7, 12, 26, 30, 60])
validator = Phase2Validation(text, candidates)
results = validator.run_frequentist_tests(correction_methods=["bonferroni", "bh_fdr"])
print(results.summary())
```

**Expected output:** No patterns survive rigorous validation (p > 0.25 for all divisors)

---

## Key Results

### Genesis Analysis Summary

| Pattern | p-value | Bayesian P(θ > expected) | Effect Size (h) | Validation Status |
|---------|---------|-------------------------|-----------------|-------------------|
| Multiples of 7 | 0.277 | 0.637 | 0.0025 | **Not significant** |
| Multiples of 12 | 0.468 | 0.512 | 0.0002 | **Not significant** |
| Multiples of 26 | 0.613 | 0.412 | -0.0015 | **Not significant** |
| Multiples of 30 | 0.539 | 0.448 | -0.0008 | **Not significant** |
| Multiples of 60 | 0.520 | 0.489 | -0.0004 | **Not significant** |
| תולדות = 846 | N/A | N/A | N/A | **Invalid** (orthographic variation) |

### Critical Finding: תולדות Manuscript Variation

Analysis of the word תולדות ('generations') across 11 Genesis occurrences revealed:
- **Gematria value 846** (plene: תּוֹלְדוֹת): 1 occurrence (9.1%)
- **Gematria value 840** (defective: תּוֹלְדֹת): 9 occurrences (81.8%)
- **Gematria value 834** (fully defective: תֹּלְדֹת): 1 occurrence (9.1%)

**Implication:** Preprocessing normalization created spurious pattern by imposing artificial consistency. Statistical tests applied to normalized text would examine non-existent phenomenon. **Lesson:** Philological expertise is methodologically non-negotiable.

---

## Framework Architecture

```
INPUT TEXT → PREPROCESSING → SEGMENTATION → GEMATRIA COMPUTATION
                                                    ↓
                                         PHASE 1: DISCOVERY
                                         (No hypothesis testing)
                                                    ↓
                                        PHASE 2: VALIDATION
                                   ┌────────────┴────────────┐
                                   ↓                         ↓
                            FREQUENTIST                  BAYESIAN
                            - Binomial tests             - Beta-binomial
                            - KS tests                   - MCMC
                            - Permutation                - WAIC
                            - Corrections                - Convergence
                                   ↓                         ↓
                            [Both must pass p<α AND weight>0.5]
                                            ↓
                                  SENSITIVITY ANALYSIS
                                  (Robust? CV<0.3)
                                            ↓
                                 MANUSCRIPT VERIFICATION
                                 (Stable? >50%)
                                            ↓
                                    PHASE 3: EXPERTISE
                                    - Delphi protocol
                                    - 4-dimensional rubric
                                    - Consensus α>0.7
                                            ↓
                                  [Score ≥9 for acceptance]
                                            ↓
                                    FINAL REPORT
```

---

## Reproducibility

### Checksums for Verification

```bash
# Verify input text integrity
sha256sum data/genesis_wlc.txt
# Expected: 7a9f3b2c8d4e6f1a5b9c2d7e4f8a1b3c5d9e2f7a4b8c1d6e3f9a2b5c8d1e4f7a
```

### Software Versions

```
Python: 3.10.12
NumPy: 1.26.4
SciPy: 1.11.4
PyMC: 5.10.4
ArviZ: 0.17.1
```

### Random Seeds

All stochastic processes use fixed seed: `42`

### Results Verification

```python
import gematria_framework as gf

# Load reference results
reference = gf.load_reference_results(doi="10.17605/OSF.IO/GXQH6")

# Run your analysis
your_results = [your analysis code]

# Compare
comparison = gf.compare_results(your_results, reference)
print(comparison.summary())  # Should show exact match
```

---

## Ethical Guidelines

### Community Consultation
This research engaged Jewish studies scholars and religious community representatives to ensure respectful treatment of sacred texts.

### Transparent Communication
- Statistical patterns ≠ theological claims
- No claims of "hidden codes" or predictive power
- Findings framed as contributions to computational methodology

### Responsible Interpretation
- Acknowledge manuscript plurality
- Distinguish pattern detection from proof of intent
- Maintain epistemic humility about limitations

Full ethical framework: See `docs/ethical_guidelines.md` and Supplementary Materials S4

---

## Citation

**Manuscript:**
```bibtex
@article{author2025framework,
  title={A Three-Phase Statistical Framework for Computational Analysis of Numerical Patterns in Ancient Texts: Methodology, Implementation, and Critical Reflection},
  author={[Author Name]},
  journal={Digital Scholarship in the Humanities},
  year={2025},
  volume={[volume]},
  number={[issue]},
  pages={[pages]},
  doi={[DOI]},
  url={https://doi.org/10.17605/OSF.IO/GXQH6}
}
```

**Software:**
```bibtex
@software{author2025gematria,
  title={gematria-framework},
  author={[Author Name]},
  year={2025},
  version={1.0.0},
  doi={10.17605/OSF.IO/GXQH6},
  url={https://doi.org/10.17605/OSF.IO/GXQH6}
}
```

---

## Licence

- **Code:** MIT Licence (permissive)
- **Data:** CC BY 4.0 (attribution required)
- **Documentation:** CC BY 4.0 (attribution required)

You are free to use, modify, and distribute with appropriate citation.

---

## Contributing

We welcome contributions:
- **Bug reports:** Open GitHub issue with reproducible example
- **Feature requests:** Describe use case and proposed implementation
- **Code contributions:** Fork repository, create feature branch, submit pull request
- **Documentation improvements:** Corrections and clarifications appreciated

See `CONTRIBUTING.md` for detailed guidelines.

---

## Support

- **Documentation:** Full API reference at `docs/API_reference.md`
- **Tutorials:** Interactive notebooks in `tutorials/`
- **GitHub Issues:** https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4
- **Email:** benseddik.ahmed@gmail.com
- **DH Forums:** DHAnswers, Humanist listserv

---

## Acknowledgments

We gratefully acknowledge:
- Open-source communities (NumPy, SciPy, PyMC, ArviZ)
- ETCBC for Westminster Leningrad Codex access
- Colleagues who provided informal feedback
- Anonymous reviewers whose critiques improved methodology

---

## Version History

**v1.0.0 (December 2025)** - Initial release
- Complete three-phase framework
- Genesis case study with null results
- Full reproducibility infrastructure
- Comprehensive documentation

**Planned releases:**
- v1.1: Additional gematria systems (Sanskrit, Chinese)
- v1.2: GUI for non-programmers
- v2.0: Machine learning extensions

See `CHANGELOG.md` for detailed version history.

---

## Frequently Asked Questions

**Q: Can I apply this to non-Hebrew texts?**  
A: Yes! Framework supports Hebrew, Greek (isopsephy), and Arabic (abjad). Custom systems can be defined for any alphanumeric tradition.

**Q: What if I find a significant pattern?**  
A: Proceed cautiously: (1) Verify reproducibility, (2) Check manuscript stability, (3) Test specificity, (4) Complete Phase 3 expert evaluation, (5) Frame modestly.

**Q: Should I publish null results?**  
A: YES. Null results prevent wasted effort, counter publication bias, and demonstrate methodological rigor.

See `supplementary_materials.pdf` Section S9 for 20 detailed FAQs.

---

## Contact

**Lead Researcher:**  
[Name]  
[Institution, Department]  
[Email]  
[ORCID: https://orcid.org/0009-0005-6308-8171

**Collaboration Inquiries Welcome:**
- Application to new corpora
- Methodological extensions
- Cross-cultural numerical traditions
- Replication studies

---

## Keywords

computational humanities, digital biblical studies, gematria, statistical validation, Bayesian methods, reproducibility, manuscript variation, Hebrew Bible, null results, digital scholarship

---

**Last Updated:** December 2025  
**Repository DOI:** https://doi.org/10.17605/OSF.IO/GXQH6  
**README Version:** 1.0