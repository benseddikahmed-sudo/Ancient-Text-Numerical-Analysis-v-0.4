[contributing_guide (3).md](https://github.com/user-attachments/files/23505197/contributing_guide.3.md)
# Contributing to Ancient Text Numerical Analysis Framework

First off, thank you for considering contributing to this project! 🎉

This is an open-source digital humanities research project, and we welcome contributions from developers, researchers, statisticians, philologists, and digital humanities practitioners.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Submission Process](#submission-process)
- [Review Process](#review-process)
- [Community](#community)

---

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of:
- Experience level (beginners welcome!)
- Background (technical or humanities)
- Identity (gender, ethnicity, religion, etc.)
- Geographic location

### Expected Behavior

- **Be respectful**: Value diverse perspectives and expertise
- **Be collaborative**: Support each other's contributions
- **Be constructive**: Offer helpful feedback, not criticism
- **Be patient**: Remember that contributors volunteer their time
- **Be culturally sensitive**: This project involves religious texts; respect all traditions

### Unacceptable Behavior

- Harassment, discrimination, or exclusionary language
- Personal attacks or trolling
- Publishing others' private information
- Misappropriation of research or code without attribution
- Misuse of framework for pseudoscience or sensationalism

### Reporting

If you experience or witness unacceptable behavior, please contact:
- **Email**: benseddik.ahmed@gmail.com
- **Subject**: "Code of Conduct Concern"

All reports will be handled confidentially.

---

## 🤝 How Can I Contribute?

### 🐛 Reporting Bugs

**Before submitting a bug report:**
- Check the [existing issues](https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/issues)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17592714.svg)](https://doi.org/10.5281/zenodo.17592714)
- Verify it's reproducible with the latest version
- Gather necessary information (see template below)

**Bug Report Template:**

```markdown
### Description
[Clear description of the bug]

### Steps to Reproduce
1. 
2. 
3. 

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Environment
- OS: [e.g., Ubuntu 20.04, macOS 13, Windows 11]
- Python version: [e.g., 3.9.7]
- Package versions: [paste output of `pip freeze`]

### Additional Context
[Screenshots, error messages, logs]
```

**Submit via**: [GitHub Issues](https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/issues/new?template=bug_report.md)

---

### 💡 Suggesting Enhancements

We welcome ideas for new features! Before submitting:

1. **Check existing suggestions**: Review [open issues with "enhancement" label](https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement)
2. **Consider scope**: Does this fit the project's mission?
3. **Think about users**: How does this help digital humanities research?

**Enhancement Template:**

```markdown
### Feature Description
[Clear description of proposed feature]

### Use Case
[Why is this valuable? Who benefits?]

### Proposed Implementation
[Optional: How might this be implemented?]

### Alternatives Considered
[Other approaches you've thought about]

### Additional Context
[Examples, mockups, references]
```

---

### 📝 Contributing Code

We accept contributions in several areas:

#### 🔬 Statistical Methods
- New validation techniques (e.g., machine learning approaches)
- Additional Bayesian models
- Non-parametric tests
- Time series analysis for sequential patterns

#### 🌍 Cultural Systems
- Additional gematria systems (Kabbalah, etc.)
- Other ancient numerical systems (Coptic, Sanskrit, etc.)
- Cross-linguistic numerical analysis

#### 📊 Visualization
- Interactive plots (Plotly, Bokeh)
- Dashboard interfaces (Streamlit, Dash)
- Publication-quality figure templates
- 3D visualizations for multi-dimensional patterns

#### 🧪 Testing
- Increase test coverage (target: >90%)
- Property-based testing (Hypothesis library)
- Integration tests
- Performance benchmarks

#### 📚 Documentation
- Tutorial notebooks
- Video walkthroughs
- API documentation improvements
- Translation to other languages

#### 🗄️ Data
- Additional manuscript transcriptions
- Variant reading compilations
- Annotated structural markers for other biblical books

---

### 📖 Contributing Documentation

Documentation is crucial for digital humanities! Areas include:

- **Tutorials**: Step-by-step guides for specific tasks
- **Methodology explanations**: Simplify complex statistical concepts
- **Use cases**: Examples from other ancient texts
- **Translations**: Non-English documentation
- **Videos**: Screencasts, explainer videos

**Documentation standards**:
- Clear, accessible language (avoid unnecessary jargon)
- Assume diverse audience (technical + humanities backgrounds)
- Include practical examples
- Cite sources appropriately

---

## 🛠️ Development Setup

### Prerequisites

- **Python 3.9+** (required)
- **Git** (required)
- **Virtual environment tool** (recommended: `venv` or `conda`)
- **Text editor/IDE** (recommended: VS Code, PyCharm, or Jupyter)

### Step 1: Fork and Clone

```bash
# Fork the repository on GitHub (click "Fork" button)

# Clone your fork
git clone https://github.com/YOUR_USERNAME/Ancient-Text-Numerical-Analysis-v-0.4.git
cd Ancient-Text-Numerical-Analysis-v-0.4

# Add upstream remote
git remote add upstream https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis-v-0.4.git
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n ancient-text python=3.9
conda activate ancient-text
```

### Step 3: Install Development Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Or manually
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Development dependencies** (`requirements-dev.txt`):
```txt
# Testing
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-xdist>=3.0.0        # Parallel testing
hypothesis>=6.0.0          # Property-based testing

# Code quality
black>=23.0.0              # Code formatter
flake8>=6.0.0              # Linter
mypy>=1.0.0                # Type checker
isort>=5.12.0              # Import sorter
pylint>=2.17.0             # Additional linting

# Documentation
sphinx>=5.0.0              # Documentation generator
sphinx-rtd-theme>=1.2.0    # ReadTheDocs theme
pdoc>=13.0.0               # API documentation

# Development tools
ipython>=8.0.0             # Enhanced Python shell
jupyter>=1.0.0             # Notebooks
pre-commit>=3.0.0          # Git hooks

# Profiling
line_profiler>=4.0.0       # Line-by-line profiling
memory_profiler>=0.61.0    # Memory usage profiling
```

### Step 4: Install Pre-commit Hooks

```bash
pre-commit install
```

This ensures code quality checks run automatically before each commit.

### Step 5: Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check code style
black --check src/ tests/
flake8 src/ tests/

# Type checking
mypy src/

# Run a simple analysis
python ancient_text_dsh.py --help
```

---

## 📂 Project Structure

Understanding the codebase:

```
Ancient-Text-Numerical-Analysis-v-0.4/
│
├── src/                              # Core source code
│   ├── __init__.py                   # Package initialization
│   ├── ancient_text_dsh.py           # Main entry point
│   ├── permutation_tests.py          # Statistical testing
│   ├── bayesian_analysis.py          # Bayesian inference
│   ├── gematria_calculator.py        # Numerical systems
│   ├── diachronic_validation.py      # Manuscript comparison
│   ├── expert_panel_analysis.py      # Delphi protocol
│   ├── fdr_correction.py             # Multiple testing
│   ├── visualization_tools.py        # Plotting utilities
│   └── utils/                        # Utility functions
│       ├── text_processing.py
│       ├── logging_config.py
│       └── config_parser.py
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures
│   ├── test_permutation.py           # Unit tests: permutation
│   ├── test_bayesian.py              # Unit tests: Bayesian
│   ├── test_gematria.py              # Unit tests: gematria
│   ├── test_fdr.py                   # Unit tests: FDR
│   ├── test_statistics.py            # Unit tests: statistics
│   ├── test_pipeline.py              # Integration tests
│   └── test_data/                    # Test fixtures
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_permutation_tests.ipynb
│   ├── 03_bayesian_validation.ipynb
│   └── ...
│
├── data/                             # Data files
│   ├── genesis_leningrad.txt
│   ├── structural_markers.json
│   └── ...
│
├── docs/                             # Documentation
│   ├── METHODOLOGY.md
│   ├── API.md
│   └── ...
│
├── results/                          # Analysis outputs
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Development dependencies
├── setup.py                          # Package setup
├── pyproject.toml                    # Build configuration
├── .pre-commit-config.yaml          # Pre-commit hooks
└── CONTRIBUTING.md                   # This file
```

### Key Modules

**Core Analysis** (`src/`):
- `permutation_tests.py`: Implements permutation testing algorithm
- `bayesian_analysis.py`: Bayes Factor calculations, MCMC
- `gematria_calculator.py`: Multi-cultural numerical systems
- `fdr_correction.py`: Benjamini-Hochberg procedure

**Utilities** (`src/utils/`):
- `text_processing.py`: Hebrew text normalization, tokenization
- `logging_config.py`: Structured logging setup
- `config_parser.py`: Configuration file handling

---

## 💻 Coding Standards

### Python Style Guide

We follow **PEP 8** with some specific conventions:

#### Code Formatting

```python
# Use Black formatter (line length: 88)
black src/ tests/

# Sort imports with isort
isort src/ tests/
```

#### Naming Conventions

```python
# Variables and functions: snake_case
def calculate_bayes_factor(observed_count, n_markers):
    pass

# Classes: PascalCase
class GematriaCalculator:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_ITERATIONS = 50000
DEFAULT_SEED = 42

# Private functions/variables: leading underscore
def _internal_helper():
    pass
```

#### Type Hints

**Always use type hints** for function signatures:

```python
from typing import List, Dict, Optional, Tuple

def permutation_test(
    corpus: List[str],
    target_term: str,
    structural_markers: List[int],
    n_iterations: int = 50000,
    seed: Optional[int] = 42
) -> Dict[str, float]:
    """
    Permutation test for lexical clustering.
    
    Parameters
    ----------
    corpus : List[str]
        Tokenized text
    target_term : str
        Target lexeme
    structural_markers : List[int]
        Marker positions
    n_iterations : int, optional
        Number of permutations (default: 50000)
    seed : int or None, optional
        Random seed (default: 42)
        
    Returns
    -------
    Dict[str, float]
        Test results with keys: 'p_value', 'observed_count', etc.
        
    Examples
    --------
    >>> result = permutation_test(corpus, 'התבה', markers)
    >>> print(result['p_value'])
    0.00974
    """
    pass
```

#### Docstrings

**Use NumPy-style docstrings**:

```python
def calculate_effect_size(observed: float, null_mean: float, null_std: float) -> float:
    """
    Calculate Cohen's d effect size.
    
    Parameters
    ----------
    observed : float
        Observed value
    null_mean : float
        Mean of null distribution
    null_std : float
        Standard deviation of null distribution
        
    Returns
    -------
    float
        Cohen's d effect size
        
    Notes
    -----
    Effect size interpretation (Cohen, 1988):
    - Small: 0.2
    - Medium: 0.5
    - Large: 0.8
    
    References
    ----------
    .. [1] Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    
    Examples
    --------
    >>> calculate_effect_size(17, 8.24, 2.07)
    4.19
    """
    if null_std == 0:
        return 0.0
    return (observed - null_mean) / null_std
```

#### Error Handling

```python
# Use specific exceptions
def load_corpus(filepath: str) -> List[str]:
    """Load text corpus from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().split()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Corpus file not found: {filepath}") from e
    except UnicodeDecodeError as e:
        raise ValueError(f"Invalid encoding in {filepath}. Expected UTF-8.") from e

# Provide informative error messages
if n_iterations < 1000:
    raise ValueError(
        f"n_iterations must be ≥1000 for reliable p-values. Got: {n_iterations}"
    )
```

#### Code Organization

```python
# Group imports logically
# Standard library
import json
import logging
from pathlib import Path
from typing import Dict, List

# Third-party
import numpy as np
import pandas as pd
from scipy import stats

# Local modules
from src.utils.text_processing import normalize_hebrew
from src.gematria_calculator import compute_gematria

# Constants at module level
DEFAULT_SIGNIFICANCE = 0.05
MAX_PERMUTATIONS = 100000

# Main code
class AnalysisPipeline:
    """Main analysis pipeline."""
    pass
```

---

## 🧪 Testing Guidelines

### Test Organization

```python
# tests/test_permutation.py

import pytest
import numpy as np
from src.permutation_tests import permutation_test

class TestPermutationTest:
    """Test suite for permutation tests."""
    
    def test_basic_functionality(self):
        """Test basic permutation test execution."""
        corpus = ['א', 'ב', 'ג', 'התבה', 'ד', 'התבה', 'ה']
        markers = [3, 5]
        
        result = permutation_test(corpus, 'התבה', markers, n_iterations=1000, seed=42)
        
        assert 'p_value' in result
        assert 0 <= result['p_value'] <= 1
        assert result['observed_count'] == 2
    
    def test_deterministic_with_seed(self):
        """Test that results are deterministic with fixed seed."""
        corpus = ['א'] * 100
        markers = [10, 20, 30]
        
        result1 = permutation_test(corpus, 'א', markers, n_iterations=1000, seed=42)
        result2 = permutation_test(corpus, 'א', markers, n_iterations=1000, seed=42)
        
        assert result1['p_value'] == result2['p_value']
    
    def test_edge_case_no_occurrences(self):
        """Test behavior when target term doesn't occur."""
        corpus = ['א', 'ב', 'ג']
        markers = [0, 1, 2]
        
        result = permutation_test(corpus, 'ד', markers, n_iterations=1000, seed=42)
        
        assert result['observed_count'] == 0
        assert result['p_value'] > 0.5  # Should be non-significant
    
    def test_invalid_input(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            permutation_test([], 'א', [0], n_iterations=100)
    
    @pytest.mark.parametrize("n_iter", [100, 1000, 10000])
    def test_different_iteration_counts(self, n_iter):
        """Test with varying iteration counts."""
        corpus = ['א'] * 50 + ['ב'] * 50
        markers = [10, 20, 30]
        
        result = permutation_test(corpus, 'א', markers, n_iterations=n_iter, seed=42)
        
        assert 'p_value' in result
        assert result['n_iterations'] == n_iter
```

### Test Fixtures

```python
# tests/conftest.py

import pytest
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_corpus():
    """Provide sample Hebrew corpus for testing."""
    return ['בראשית', 'ברא', 'אלהים', 'את', 'השמים', 'ואת', 'הארץ']

@pytest.fixture
def sample_markers():
    """Provide sample structural markers."""
    return [0, 3, 6]

@pytest.fixture
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / 'test_data'

@pytest.fixture
def mock_bayesian_model(mocker):
    """Mock PyMC model for fast testing."""
    mock = mocker.patch('src.bayesian_analysis.pm.sample')
    mock.return_value = np.random.randn(1000, 2)  # Mock posterior samples
    return mock
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_permutation.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast tests (skip slow Bayesian tests)
pytest tests/ -m "not slow"

# Run tests in parallel (faster)
pytest tests/ -n auto

# Run with verbose output
pytest tests/ -vv -s
```

### Test Markers

```python
# Mark slow tests
@pytest.mark.slow
def test_full_bayesian_analysis():
    """This test takes ~5 minutes."""
    pass

# Mark integration tests
@pytest.mark.integration
def test_complete_pipeline():
    """End-to-end pipeline test."""
    pass

# Mark tests requiring external data
@pytest.mark.requires_data
def test_with_manuscript_data():
    """Requires downloaded manuscript files."""
    pass
```

### Coverage Goals

- **Minimum**: 85% overall coverage
- **Target**: 90%+ for core modules
- **Critical modules**: 95%+ (permutation_tests, bayesian_analysis)

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # View in browser
```

---

## 📚 Documentation Standards

### Module Docstrings

```python
"""
Permutation Testing Module
==========================

This module implements permutation tests for lexical clustering analysis
in ancient texts.

The permutation test provides an exact, assumption-free method for assessing
statistical significance by comparing observed data to a null distribution
generated through random permutations.

Key Functions
-------------
permutation_test : Main permutation testing function
bootstrap_ci : Bootstrap confidence intervals
calculate_effect_size : Cohen's d effect size

Examples
--------
>>> from src.permutation_tests import permutation_test
>>> result = permutation_test(corpus, 'התבה', markers, n_iterations=10000)
>>> print(f"P-value: {result['p_value']:.4f}")
P-value: 0.0097

See Also
--------
bayesian_analysis : Bayesian validation methods
fdr_correction : Multiple testing correction

References
----------
.. [1] Good, P. I. (2005). Permutation, Parametric, and Bootstrap Tests 
       of Hypotheses. Springer.
"""
```

### Function Documentation

**Required elements**:
1. Short description (one line)
2. Extended description (if needed)
3. Parameters with types
4. Returns with types
5. Raises (exceptions)
6. Examples
7. Notes (optional)
8. References (if applicable)

### Inline Comments

```python
def permutation_test(corpus, target_term, markers, n_iterations=50000, seed=42):
    """Permutation test for lexical clustering."""
    
    np.random.seed(seed)
    
    # Observed count at structural markers
    # We count exact matches only (no substring matching)
    observed_count = sum(
        1 for idx in markers
        if corpus[idx] == target_term
    )
    
    # Generate null distribution via permutation
    # This preserves lexical frequencies while randomizing positions
    null_distribution = []
    
    for i in range(n_iterations):
        # Shuffle entire corpus
        shuffled_corpus = np.random.permutation(corpus)
        
        # Count in shuffled version
        shuffled_count = sum(
            1 for idx in markers
            if shuffled_corpus[idx] == target_term
        )
        
        null_distribution.append(shuffled_count)
    
    # Calculate one-tailed p-value
    # We use ≥ because we're testing for enrichment
    null_distribution = np.array(null_distribution)
    p_value = np.mean(null_distribution >= observed_count)
    
    return {
        'p_value': p_value,
        'observed_count': observed_count,
        'null_distribution': null_distribution
    }
```

### README Updates

When adding features, update:
- **README.md**: User-facing documentation
- **CHANGELOG.md**: Version history
- **docs/API.md**: API reference (if applicable)

---

## 🔄 Submission Process

### 1. Create a Feature Branch

```bash
# Ensure your main branch is up to date
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/my-awesome-feature

# Or for bug fixes
git checkout -b fix/bug-description
```

### 2. Make Your Changes

```bash
# Edit files
# Write tests
# Update documentation

# Check status
git status

# Add files
git add src/my_new_file.py tests/test_my_new_file.py

# Commit with clear message
git commit -m "Add gematria support for Greek isopsephy

- Implement Greek letter-to-number mapping
- Add unit tests with 95% coverage
- Update documentation with examples
- Closes #42"
```

### Commit Message Guidelines

**Format**:
```
<type>: <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Example**:
```
feat: Add Arabic abjad gematria system

- Implement Arabic letter mappings (أ=1, ب=2, etc.)
- Add comprehensive unit tests (97% coverage)
- Include examples in documentation
- Support for both standalone and contextual forms

Closes #127
```

### 3. Run Pre-commit Checks

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/
pylint src/

# Type check
mypy src/

# Run tests
pytest tests/ -v --cov=src

# All checks pass? Great!
```

### 4. Push to Your Fork

```bash
git push origin feature/my-awesome-feature
```

### 5. Create Pull Request

1. Go to GitHub repository
2. Click "New Pull Request"
3. Select your branch
4. Fill out PR template (see below)
5. Submit!

**Pull Request Template**:

```markdown
## Description
[Brief description of changes]

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Motivation and Context
[Why is this change needed? What problem does it solve?]

## How Has This Been Tested?
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

Test Configuration:
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]

## Screenshots (if applicable)
[Add screenshots for UI changes]

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review performed
- [ ] Code commented where necessary
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings
- [ ] CHANGELOG.md updated

## Related Issues
Closes #[issue number]
```

---

## 👀 Review Process

### What Reviewers Look For

1. **Code Quality**
   - Follows PEP 8 and project conventions
   - Clear, readable code
   - Appropriate comments and docstrings

2. **Testing**
   - Comprehensive test coverage
   - Edge cases handled
   - Tests actually test what they claim

3. **Documentation**
   - README updated if needed
   - API documentation complete
   - Examples provided

4. **Functionality**
   - Feature works as described
   - No breaking changes (or clearly documented)
   - Integrates well with existing code

5. **Performance**
   - No significant performance regression
   - Efficient algorithms used
   - Memory usage reasonable

### Review Timeline

- **Initial review**: Within 3-5 days
- **Subsequent reviews**: Within 2 days
- **Small PRs** (<100 lines): Faster review
- **Large PRs** (>500 lines): May take longer (consider splitting)

### Addressing Review Comments

```bash
# Make requested changes
git add changed_files
git commit -m "Address review comments: fix type hints"

# Push updates
git push origin feature/my-awesome-feature

# PR automatically updates!
```

### After Approval

Once approved, a maintainer will:
1. Squash and merge your PR
2. Update CHANGELOG.md
3. Thank you for your contribution! 🎉

---

## 🌟 Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, general discussion
- **Email**: benseddik.ahmed@gmail.com (for sensitive matters)

### Recognition

Contributors are recognized in:
- **README.md**: Contributors section
- **CHANGELOG.md**: Version credits
- **Academic papers**: Acknowledgments section (with permission)

### Levels of Contribution

**🥉 Contributor**: Any accepted PR  
**🥈 Regular Contributor**: 3+ accepted PRs  
**🥇 Core Contributor**: 10+ PRs or major feature  
**💎 Maintainer**: Invited role with commit access

---

## 📖 Additional Resources

### Learning Resources

**Python & Testing**:
- [Real Python](https://realpython.com/)
- [pytest documentation](https://docs.pytest.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

**Digital Humanities**:
- [Programming Historian](https://programminghistorian.org/)
- [Digital Humanities Quarterly](http://digitalhumanities.org/dhq/)

**Statistical Methods**:
- [Think Stats](https://greenteapress.com/thinkstats2/html/index.html)
- [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/)

**Biblical Studies**:
- [Open Scriptures](https://github.com/openscriptures)
- [ETCBC](https://github.com/ETCBC)

### Related Projects

- [Hebrew NLP](https://github.com/NLPH/NLPH_Resources)
- [PyMC Examples](https://www.pymc.io/projects/examples/)
- [Digital Biblical Studies](https://github.com/topics/biblical-studies)

---

## 🎓 First-Time Contributors

Never contributed to open source before? No problem!

### Good First Issues

Look for issues labeled:
- `good first issue`: Beginner-friendly
- `help wanted`: Extra attention needed
- `documentation`: Non-code contribution

### Getting Help

Stuck? Ask for help:
- Comment on the issue
- Open a GitHub Discussion
- Email the maintainer

**Remember**: Everyone was a beginner once. We're here to help! 💪

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Thank You!

Your contributions make this project better for the entire digital humanities community. Whether you're fixing a typo, adding a feature, or helping with documentation—every contribution matters.

**Happy coding!** 🚀

---

**Last Updated**: November 2025  
**Maintained by**: Ahmed Benseddik  
**Questions?** Open an issue or email benseddik.ahmed@gmail.com
