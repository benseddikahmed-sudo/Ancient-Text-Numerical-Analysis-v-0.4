[dsh_readme.md](https://github.com/benseddikahmed-attachments/files/23151779/dsh_readme.md)
# Ancient Text Numerical Analysis Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
["https://doi.org/10.5281/zenodo.17443361"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17443361.svg" alt="DOI"></a>

A rigorous, reproducible framework for computational analysis of numerical patterns in ancient texts using multiple cultural systems (Hebrew, Greek, Arabic) with comprehensive statistical validation and ethical considerations.

**Publication**: *Digital Scholarship in the Humanities* (DSH)  
**Author**: Ahmed Benseddik  
**Version**: 4.0-DSH  
**Date**: 2025-10-26

---

## 🎯 Key Features

### Statistical Rigor
- **Frequentist methods**: Binomial tests with Bonferroni/Šidák corrections
- **Bayesian inference**: Hierarchical models with WAIC comparison
- **Non-parametric validation**: Permutation tests (10,000+ iterations)
- **Power analysis**: Sample size adequacy assessment
- **Effect sizes**: Cohen's h, standardized differences
- **Confidence intervals**: Wilson score, HDI (Highest Density Interval)

### Reproducibility
- Complete environment capture (Python version, dependencies, system info)
- Git commit tracking
- Deterministic random seeds
- Comprehensive logging (file + console)
- Metadata for every analysis run

### Cultural Systems
- Hebrew Gematria (standard, Atbash, Albam)
- Greek Isopsephy
- Arabic Abjad numerals
- Cross-cultural correlation analysis

### Visualizations
- Publication-quality figures (300 DPI)
- Distribution histograms with Q-Q plots
- Bayesian posterior distributions
- Sensitivity analysis plots
- Cross-cultural heatmaps

### Ethical Framework
- Methodological transparency
- Cultural sensitivity guidelines
- Interpretation caveats
- Acknowledgment of limitations

---

## 📋 Requirements

### Minimum Requirements
- Python 3.9+
- NumPy >= 1.24
- SciPy >= 1.10
- Pandas >= 2.0
- Matplotlib >= 3.7

### Recommended (for full functionality)
- PyMC >= 5.0 (Bayesian analysis)
- ArviZ >= 0.15 (Bayesian diagnostics)
- Numba >= 0.57 (performance optimization)
- Seaborn >= 0.12 (enhanced visualizations)

Install all dependencies:
```bash
pip install -r requirements.txt
```

For minimal installation (without Bayesian):
```bash
pip install numpy scipy pandas matplotlib seaborn
```

---

## 🚀 Quick Start

### Basic Usage

```bash
# Run complete analysis
python ancient_text_dsh.py --data-dir ./data --output-dir ./results

# Fast analysis (no Bayesian)
python ancient_text_dsh.py --no-bayesian

# High-quality analysis
python ancient_text_dsh.py --n-permutations 50000 --n-bayesian-draws 5000
```

### Input Data Format

Place your text file in the data directory:
```
data/
  └── text.txt  # Hebrew text, UTF-8 encoded
```

The framework automatically:
- Validates character encoding
- Normalizes final letter forms
- Extracts word windows
- Computes numerical values

### Output Structure

```
output/
  ├── results_YYYYMMDD_HHMMSS.json      # Complete results (JSON)
  ├── report_YYYYMMDD_HHMMSS.md         # Human-readable report
  ├── analysis.log                       # Detailed log file
  ├── figures/
  │   ├── gematria_distribution.png
  │   ├── multiples_analysis.png
  │   ├── cross_cultural_heatmap.png
  │   ├── sensitivity_analysis.png
  │   └── posterior_divisor_*.png
  └── tables/
      └── multiples_frequentist_*.csv
```

---

## 📊 Analysis Pipeline

### 1. Gematria Analysis
- Compute numerical values for text segments
- Statistical summaries (mean, median, quartiles)
- Distribution testing (normality checks)
- Cross-cultural comparison

### 2. Multiples Analysis (Frequentist)
- Test enrichment of specific divisors (7, 12, 26, 30, 60)
- Binomial tests with exact p-values
- Multiple testing corrections (Bonferroni, Šidák)
- Effect sizes and confidence intervals

### 3. Bayesian Hierarchical Modeling
- Compare null vs. enrichment hypotheses
- MCMC sampling (4 chains, convergence diagnostics)
- Model comparison (WAIC/LOO)
- Posterior predictive checks

### 4. Sensitivity Analysis
- Test robustness to window size
- Sampling strategy variations
- Parameter stability assessment

### 5. Validation Suite
- Distribution tests (Shapiro-Wilk, Anderson-Darling)
- Sample size adequacy
- Power analysis
- Assumption checking

---

## 🔧 Advanced Usage

### Custom Configuration

```python
from ancient_text_dsh import AnalysisConfig, AncientTextAnalysisPipeline

config = AnalysisConfig(
    data_dir='custom/path',
    output_dir='custom/output',
    random_seed=123,
    n_permutations=20000,
    n_bayesian_draws=5000,
    enable_bayesian=True,
    significance_level=0.01
)

pipeline = AncientTextAnalysisPipeline(config)
results = pipeline.run_complete_analysis()
```

### Programmatic Access

```python
from ancient_text_dsh import compute_gematria, CulturalSystem

# Compute values
hebrew_value = compute_gematria('בראשית', CulturalSystem.HEBREW_STANDARD)
greek_value = compute_gematria('λόγος', CulturalSystem.GREEK_ISOPSEPHY)
arabic_value = compute_gematria('بسم', CulturalSystem.ARABIC_ABJAD)

print(f"Hebrew: {hebrew_value}")  # 913
print(f"Greek: {greek_value}")
print(f"Arabic: {arabic_value}")
```

### Batch Processing

```python
from pathlib import Path

data_files = Path('corpus').glob('*.txt')

for file in data_files:
    config = AnalysisConfig(
        data_dir=file.parent,
        output_dir=Path('results') / file.stem
    )
    pipeline = AncientTextAnalysisPipeline(config)
    pipeline.run_complete_analysis()
```

---

## 📈 Interpreting Results

### Statistical Significance
- **p < 0.05**: Statistically significant (after corrections)
- **Effect size (h)**: 0.2 (small), 0.5 (medium), 0.8 (large)
- **Bayesian**: ΔWAIC > 6 indicates strong evidence

### Sensitivity Analysis
- **CV < 0.3**: Robust results
- **0.3 < CV < 0.5**: Moderate sensitivity
- **CV > 0.5**: High sensitivity, interpret with caution

### Power Analysis
- **Power > 0.8**: Adequate sample size
- **Power < 0.8**: Consider larger sample or effect size

---

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=ancient_text_dsh
```

Generate coverage report:
```bash
pytest tests/ --cov=ancient_text_dsh --cov-report=html
```

Run specific test categories:
```bash
# Unit tests only
pytest tests/test_gematria.py

# Statistical tests
pytest tests/test_statistics.py

# Integration tests
pytest tests/test_pipeline.py
```

---

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@article{benseddik2025ancient,
  title={Ancient Text Numerical Analysis: A Statistical Framework with Ethical Considerations},
  author={Benseddik, Ahmed},
  journal={Digital Scholarship in the Humanities},
  year={2025},
  publisher={Oxford University Press},
  doi=10.5281/zenodo.17443361
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4).git
cd ancient-text-analysis
pip install -e ".[dev]"
pre-commit install
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Traditional scholarship communities for cultural guidance
- PyMC developers for Bayesian modeling tools
- Digital humanities community for methodological discussions

---

## 📞 Contact

**Ahmed Benseddik**  
Email: benseddik.ahmed@gmail.com  
GitHub: [@ahmedbenseddik](https://github.com/ahmedbenseddik)

---

## 🔗 Links

- **Documentation**: https://ancient-text-analysis.readthedocs.io
- **Repository**:https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4
- **Issue Tracker**: https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/issues
- **DSH Journal**: https://academic.oup.com/dsh

---

## 📚 References

### Methodology
- Efron, B., & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
- Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.

### Cultural Systems
- Blech, B. (2004). *The Complete Idiot's Guide to Jewish Culture*. Alpha Books.
- Ifrah, G. (2000). *The Universal History of Numbers*. Wiley.

### Digital Humanities
- Schöch, C. (2013). Big? Smart? Clean? Messy? Data in the Humanities. *Journal of Digital Humanities*, 2(3).

---

*Last updated: 2025-10-26*
